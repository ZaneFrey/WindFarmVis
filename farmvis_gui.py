"""Native desktop driver for the ERF wind-farm domain visualizer.

Run from the farmvis environment:
    python farmvis_gui.py
"""

from __future__ import annotations

import html
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6 import QtCore, QtWidgets
from pyvistaqt import BackgroundPlotter

from farmvis_tool import (
    AMRLevelConfig,
    ClusterConfig,
    DomainConfig,
    FarmStudyConfig,
    compute_amr_metrics,
    compute_grid_metrics,
    compute_vertical_grid_profile,
    configs_from_preset,
    export_cluster_coordinates,
    export_vertical_grid_levels,
    generate_all_layouts,
    list_layout_presets,
    load_layout_preset,
    plot_xy,
    plot_xz,
    plot_yz,
    populate_3d_scene,
    save_layout_preset,
    validate_configuration,
)


TOOL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TOOL_DIR / "outputs"
PRESET_DIR = TOOL_DIR / "presets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRESET_DIR.mkdir(parents=True, exist_ok=True)

AMR_COLORS = ["red", "orange", "gold", "green", "deepskyblue", "blue", "magenta"]
TURBINE_PRESETS = (
    ("NREL 5 MW", 126.0, 90.0),
    ("DTU 10 MW", 178.3, 119.0),
    ("IEA 10 MW", 198.0, 119.0),
    ("IEA 15 MW", 240.0, 150.0),
    ("Vestas v236", 236.0, 115.0),
    ("Siemens Gamesa SG 11.0-200 DD", 200.0, 140.0),
)
DEFAULT_TURBINE_PRESET = "IEA 15 MW"


class OneDecimalDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """Keep full input precision while presenting compact values in the panel."""

    def textFromValue(self, value: float) -> str:
        return f"{value:.1f}"


class TwoDWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FarmVis 2D Views")
        self.resize(1050, 760)
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self.canvases: dict[str, FigureCanvasQTAgg] = {}
        for label in ("x-y", "x-z", "y-z"):
            canvas = FigureCanvasQTAgg(plt.Figure())
            self.tabs.addTab(canvas, label)
            self.canvases[label] = canvas

    def update_figures(self, domain: DomainConfig, study: FarmStudyConfig, layouts: dict, amr: dict) -> None:
        figures = {
            "x-y": plot_xy(domain, study, layouts, amr_metrics=amr),
            "x-z": plot_xz(domain, study, layouts, amr_metrics=amr),
            "y-z": plot_yz(domain, study, layouts, amr_metrics=amr),
        }
        for label, new_figure in figures.items():
            old_canvas = self.canvases[label]
            index = self.tabs.indexOf(old_canvas)
            self.tabs.removeTab(index)
            old_canvas.setParent(None)
            plt.close(old_canvas.figure)
            canvas = FigureCanvasQTAgg(new_figure)
            self.tabs.insertTab(index, canvas, label)
            self.canvases[label] = canvas


class VerticalGridWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vertical Grid Discretization")
        self.resize(720, 760)
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        self.canvas = FigureCanvasQTAgg(plt.Figure())
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setCentralWidget(container)

    def update_profile(self, domain: DomainConfig) -> None:
        profile = compute_vertical_grid_profile(domain)
        cell_sizes = profile["cell_sizes"]
        cell_centers = profile["cell_centers"]
        z_levels = profile["z_levels"]
        x_max = float(np.max(cell_sizes))
        x_limit = x_max * 1.05 if x_max > 0.0 else 1.0

        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.hlines(
            z_levels,
            xmin=0.0,
            xmax=x_limit,
            colors="lightgray",
            linewidth=0.6,
            zorder=0,
        )
        ax.plot(cell_sizes, cell_centers, color="black", linewidth=2.0, zorder=2)
        ax.set_xlabel(r"$\Delta z$ (m)")
        ax.set_ylabel(r"$z$ (m)")
        ax.set_title(f"{profile['type'].replace('_', ' ').title()} Vertical Grid")
        ax.set_xlim(0.0, x_limit)
        ax.set_ylim(0.0, domain.Lz)
        ax.set_axisbelow(True)
        self.canvas.figure.tight_layout()
        self.canvas.draw_idle()

class FarmVisWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ERF Wind Farm Domain Visualizer")
        self.resize(760, 940)
        self.cluster_editors: list[dict] = []
        self.amr_editors: list[dict] = []
        self.plotter: BackgroundPlotter | None = None
        self._rotor_angle_signature: tuple | None = None
        self._turbine_signature: tuple | None = None
        self._rotor_angles: list[np.ndarray] | None = None
        self.two_d_window = TwoDWindow()
        self.vertical_grid_window: VerticalGridWindow | None = None

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        contents = QtWidgets.QWidget()
        self.layout = QtWidgets.QVBoxLayout(contents)
        self.layout.setContentsMargins(12, 12, 12, 12)
        self.layout.setSpacing(10)
        scroll.setWidget(contents)
        self.setCentralWidget(scroll)

        self.panel_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.panel_splitter.setChildrenCollapsible(False)
        self.panel_splitter.setHandleWidth(10)
        self.panel_splitter.setStyleSheet(
            "QSplitter::handle:vertical {"
            "background: palette(midlight);"
            "border-top: 1px solid palette(mid);"
            "border-bottom: 1px solid palette(mid);"
            "margin: 3px 0;"
            "}"
            "QSplitter::handle:vertical:hover { background: palette(mid); }"
        )
        self.layout.addWidget(self.panel_splitter)

        self.control_tabs = QtWidgets.QTabWidget()
        self.panel_splitter.addWidget(self.control_tabs)
        self._build_domain_controls()
        self._build_farm_controls()
        self._build_amr_controls()
        self._build_preset_controls()
        self._build_output_controls()
        default_tab_height = round(self.control_tabs.sizeHint().height() * 1.3)
        self.panel_splitter.setSizes([default_tab_height, self.output_panel.sizeHint().height()])
        self.panel_splitter.handle(1).setToolTip("Drag to resize the control tabs")
        self._refresh_preset_list()
        self.update_visualization()

    def _control_tab(self, title: str) -> QtWidgets.QVBoxLayout:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        self.control_tabs.addTab(page, title)
        return layout

    @staticmethod
    def _float(value: float, minimum: float = -1e9, maximum: float = 1e9, step: float = 1.0) -> QtWidgets.QDoubleSpinBox:
        widget = OneDecimalDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(6)
        widget.setSingleStep(step)
        widget.setValue(value)
        return widget

    @staticmethod
    def _int(value: int, minimum: int = 0, maximum: int = 10_000_000) -> QtWidgets.QSpinBox:
        widget = QtWidgets.QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        return widget

    @staticmethod
    def _form_row(*widgets: tuple[str, QtWidgets.QWidget]) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        for label, widget in widgets:
            layout.addWidget(QtWidgets.QLabel(label))
            layout.addWidget(widget)
        layout.addStretch(1)
        return row

    def _build_domain_controls(self) -> None:
        layout = self._control_tab("Domain")
        self.lx = self._float(12000.0, minimum=0.001, step=100.0)
        self.ly = self._float(6000.0, minimum=0.001, step=100.0)
        self.lz = self._float(1500.0, minimum=0.001, step=50.0)
        layout.addWidget(self._form_row(("L_x (m)", self.lx), ("L_y (m)", self.ly), ("L_z (m)", self.lz)))

        self.grid_mode = QtWidgets.QComboBox()
        self.grid_mode.addItem("Resolution input", "resolution")
        self.grid_mode.addItem("Grid counts", "counts")
        layout.addWidget(self._form_row(("Grid mode", self.grid_mode)))
        self.grid_stack = QtWidgets.QStackedWidget()
        resolution_page = QtWidgets.QWidget()
        resolution_layout = QtWidgets.QHBoxLayout(resolution_page)
        resolution_layout.setContentsMargins(0, 0, 0, 0)
        self.dx = self._float(20.0, minimum=0.001)
        self.dy = self._float(20.0, minimum=0.001)
        resolution_layout.addWidget(self._form_row(("d_x (m)", self.dx), ("d_y (m)", self.dy)))
        count_page = QtWidgets.QWidget()
        count_layout = QtWidgets.QHBoxLayout(count_page)
        count_layout.setContentsMargins(0, 0, 0, 0)
        self.nx = self._int(600, minimum=1)
        self.ny = self._int(300, minimum=1)
        self.nz = self._int(150, minimum=1)
        count_layout.addWidget(self._form_row(("N_x", self.nx), ("N_y", self.ny)))
        self.grid_stack.addWidget(resolution_page)
        self.grid_stack.addWidget(count_page)
        layout.addWidget(self.grid_stack)
        self.grid_mode.currentIndexChanged.connect(self.grid_stack.setCurrentIndex)

        self.vertical_discretization = QtWidgets.QComboBox()
        self.vertical_discretization.addItem("Uniform", "uniform")
        self.vertical_discretization.addItem("Stretched", "stretched")
        self.vertical_discretization.addItem("Hyperbolic tangent", "hyperbolic_tangent")
        self.vertical_discretization.addItem("Piecewise", "piecewise")
        layout.addWidget(self._form_row(("Vertical discretization", self.vertical_discretization)))

        self.dz = self._float(10.0, minimum=0.001)
        self.initial_dz = self._float(10.0, minimum=0.001)
        self.stretching_ratio = QtWidgets.QDoubleSpinBox()
        self.stretching_ratio.setRange(1.000001, 10.0)
        self.stretching_ratio.setDecimals(6)
        self.stretching_ratio.setSingleStep(0.01)
        self.stretching_ratio.setValue(1.03)
        self.stretched_final_dz = self._float(50.0, minimum=0.001)
        self.stretched_parameter_mode = QtWidgets.QComboBox()
        self.stretched_parameter_mode.addItem("Stretching ratio", "ratio")
        self.stretched_parameter_mode.addItem("Final d_z", "final_dz")
        self.uniform_dz_row = self._form_row(("d_z (m)", self.dz))
        self.vertical_nz_row = self._form_row(("N_z", self.nz))
        self.stretched_initial_row = self._form_row(("initial d_z (m)", self.initial_dz))
        self.stretched_mode_row = self._form_row(("Specify stretching with", self.stretched_parameter_mode))
        self.stretched_ratio_row = self._form_row(("stretching ratio", self.stretching_ratio))
        self.stretched_final_row = self._form_row(("final d_z (m)", self.stretched_final_dz))
        self.tanh_initial_dz = self._float(10.0, minimum=0.001)
        self.tanh_stretching_factor = QtWidgets.QDoubleSpinBox()
        self.tanh_stretching_factor.setRange(0.000001, 10.0)
        self.tanh_stretching_factor.setDecimals(6)
        self.tanh_stretching_factor.setSingleStep(0.1)
        self.tanh_stretching_factor.setValue(2.0)
        self.tanh_stretching_factor.setToolTip(
            "Larger values concentrate more vertical cells near the ground."
        )
        self.tanh_final_dz = self._float(50.0, minimum=0.001)
        self.tanh_parameter_mode = QtWidgets.QComboBox()
        self.tanh_parameter_mode.addItem("Stretching factor", "factor")
        self.tanh_parameter_mode.addItem("Final d_z", "final_dz")
        self.tanh_initial_row = self._form_row(("initial d_z (m)", self.tanh_initial_dz))
        self.tanh_mode_row = self._form_row(("Specify stretching with", self.tanh_parameter_mode))
        self.tanh_factor_row = self._form_row(("stretching factor", self.tanh_stretching_factor))
        self.tanh_final_row = self._form_row(("final d_z (m)", self.tanh_final_dz))
        self.piecewise_uniform_dz = self._float(10.0, minimum=0.001)
        self.piecewise_transition_height = self._float(300.0, minimum=0.001, step=25.0)
        self.piecewise_stretching_ratio = QtWidgets.QDoubleSpinBox()
        self.piecewise_stretching_ratio.setRange(1.000001, 10.0)
        self.piecewise_stretching_ratio.setDecimals(6)
        self.piecewise_stretching_ratio.setSingleStep(0.01)
        self.piecewise_stretching_ratio.setValue(1.03)
        self.piecewise_final_dz = self._float(50.0, minimum=0.001)
        self.piecewise_parameter_mode = QtWidgets.QComboBox()
        self.piecewise_parameter_mode.addItem("Stretching ratio", "ratio")
        self.piecewise_parameter_mode.addItem("Final d_z", "final_dz")
        self.piecewise_base_row = self._form_row(
            ("uniform d_z (m)", self.piecewise_uniform_dz),
            ("z_t (m)", self.piecewise_transition_height),
        )
        self.piecewise_mode_row = self._form_row(
            ("Specify stretching with", self.piecewise_parameter_mode)
        )
        self.piecewise_ratio_row = self._form_row(
            ("stretching ratio", self.piecewise_stretching_ratio)
        )
        self.piecewise_final_row = self._form_row(("final d_z (m)", self.piecewise_final_dz))
        layout.addWidget(self.uniform_dz_row)
        layout.addWidget(self.vertical_nz_row)
        layout.addWidget(self.stretched_initial_row)
        layout.addWidget(self.stretched_mode_row)
        layout.addWidget(self.stretched_ratio_row)
        layout.addWidget(self.stretched_final_row)
        layout.addWidget(self.tanh_initial_row)
        layout.addWidget(self.tanh_mode_row)
        layout.addWidget(self.tanh_factor_row)
        layout.addWidget(self.tanh_final_row)
        layout.addWidget(self.piecewise_base_row)
        layout.addWidget(self.piecewise_mode_row)
        layout.addWidget(self.piecewise_ratio_row)
        layout.addWidget(self.piecewise_final_row)
        self.grid_mode.currentIndexChanged.connect(self._update_vertical_controls)
        self.vertical_discretization.currentIndexChanged.connect(self._update_vertical_controls)
        self.stretched_parameter_mode.currentIndexChanged.connect(self._update_vertical_controls)
        self.tanh_parameter_mode.currentIndexChanged.connect(self._update_vertical_controls)
        self.piecewise_parameter_mode.currentIndexChanged.connect(self._update_vertical_controls)
        self._update_vertical_controls()

        self.inflow_enabled = QtWidgets.QCheckBox("Enable inflow region")
        self.inflow_enabled.setChecked(True)
        self.lin = self._float(2000.0, minimum=0.0, step=100.0)
        self.rayleigh_enabled = QtWidgets.QCheckBox("Enable Rayleigh region")
        self.rayleigh_enabled.setChecked(True)
        self.rayleigh_depth = self._float(300.0, minimum=0.0, step=25.0)
        layout.addWidget(self._form_row(("", self.inflow_enabled), ("L_in (m)", self.lin)))
        layout.addWidget(self._form_row(("", self.rayleigh_enabled), ("Depth (m)", self.rayleigh_depth)))
        self.inflow_enabled.toggled.connect(lambda enabled: self.lin.setEnabled(enabled))
        self.rayleigh_enabled.toggled.connect(lambda enabled: self.rayleigh_depth.setEnabled(enabled))

    def _build_amr_controls(self) -> None:
        layout = self._control_tab("AMR")
        self.max_level = self._int(0, minimum=0, maximum=10)
        layout.addWidget(self._form_row(("Maximum level", self.max_level)))
        self.amr_toolbox = QtWidgets.QToolBox()
        layout.addWidget(self.amr_toolbox)
        self.max_level.valueChanged.connect(self._rebuild_amr_editors)
        self._update_amr_availability()

    def _update_vertical_controls(self, _index: int | None = None) -> None:
        vertical_type = self.vertical_discretization.currentData()
        counts_mode = self.grid_mode.currentData() == "counts"
        self.uniform_dz_row.setVisible(vertical_type == "uniform" and not counts_mode)
        self.vertical_nz_row.setVisible(counts_mode and vertical_type != "piecewise")
        stretched = vertical_type == "stretched"
        self.stretched_initial_row.setVisible(stretched)
        self.stretched_mode_row.setVisible(stretched)
        self.stretched_ratio_row.setVisible(
            stretched and self.stretched_parameter_mode.currentData() == "ratio"
        )
        self.stretched_final_row.setVisible(
            stretched and self.stretched_parameter_mode.currentData() == "final_dz"
        )
        self.tanh_initial_row.setVisible(vertical_type == "hyperbolic_tangent" and not counts_mode)
        tanh = vertical_type == "hyperbolic_tangent"
        self.tanh_mode_row.setVisible(tanh)
        self.tanh_factor_row.setVisible(tanh and self.tanh_parameter_mode.currentData() == "factor")
        self.tanh_final_row.setVisible(tanh and self.tanh_parameter_mode.currentData() == "final_dz")
        piecewise = vertical_type == "piecewise"
        self.piecewise_base_row.setVisible(piecewise)
        self.piecewise_mode_row.setVisible(piecewise)
        self.piecewise_ratio_row.setVisible(
            piecewise and self.piecewise_parameter_mode.currentData() == "ratio"
        )
        self.piecewise_final_row.setVisible(
            piecewise and self.piecewise_parameter_mode.currentData() == "final_dz"
        )
        self._update_amr_availability()

    def _update_amr_availability(self) -> None:
        if not hasattr(self, "max_level") or not hasattr(self, "vertical_discretization"):
            return
        uniform = self.vertical_discretization.currentData() == "uniform"
        if not uniform:
            self.max_level.setValue(0)
        self.max_level.setEnabled(uniform)
        self.amr_toolbox.setEnabled(uniform)
        self.max_level.setToolTip("" if uniform else "AMR is unavailable for non-uniform vertical grids.")

    def _build_farm_controls(self) -> None:
        layout = self._control_tab("Wind Farm")
        self.turbine_preset = QtWidgets.QComboBox()
        for name, rotor_diameter, hub_height in TURBINE_PRESETS:
            self.turbine_preset.addItem(name, (rotor_diameter, hub_height))
        self.turbine_preset.setCurrentText(DEFAULT_TURBINE_PRESET)
        self.turbine_preset.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.turbine_preset.setMinimumContentsLength(14)
        self.turbine_preset.setMaximumWidth(260)

        default_rotor_diameter, default_hub_height = self.turbine_preset.currentData()
        self.rotor_diameter = self._float(default_rotor_diameter, minimum=0.001, step=5.0)
        self.hub_height = self._float(default_hub_height, minimum=0.001, step=5.0)
        self.cluster_count = self._int(1, minimum=1, maximum=20)
        layout.addWidget(self._form_row(("Rotor D (m)", self.rotor_diameter), ("Hub height (m)", self.hub_height)))
        layout.addWidget(self._form_row(("# clusters", self.cluster_count), ("Preset", self.turbine_preset)))
        self.cluster_toolbox = QtWidgets.QToolBox()
        layout.addWidget(self.cluster_toolbox)
        self.turbine_preset.currentIndexChanged.connect(self._apply_turbine_preset)
        self.cluster_count.valueChanged.connect(self._rebuild_cluster_editors)
        self._rebuild_cluster_editors(1)

    def _apply_turbine_preset(self, index: int) -> None:
        dimensions = self.turbine_preset.itemData(index)
        if dimensions is None:
            return
        rotor_diameter, hub_height = dimensions
        self.rotor_diameter.setValue(rotor_diameter)
        self.hub_height.setValue(hub_height)

    def _sync_turbine_preset(self, rotor_diameter: float, hub_height: float) -> None:
        matching_index = -1
        for index in range(self.turbine_preset.count()):
            preset_rotor, preset_hub = self.turbine_preset.itemData(index)
            if np.isclose(rotor_diameter, preset_rotor) and np.isclose(hub_height, preset_hub):
                matching_index = index
                break
        with QtCore.QSignalBlocker(self.turbine_preset):
            self.turbine_preset.setCurrentIndex(matching_index)

    def _build_preset_controls(self) -> None:
        layout = self._control_tab("Presets")
        self.preset_name = QtWidgets.QLineEdit("default_layout")
        self.preset_combo = QtWidgets.QComboBox()
        save_button = QtWidgets.QPushButton("Save Layout")
        load_button = QtWidgets.QPushButton("Load Layout")
        refresh_button = QtWidgets.QPushButton("Refresh")
        save_button.clicked.connect(self.save_preset)
        load_button.clicked.connect(self.load_preset)
        refresh_button.clicked.connect(self._refresh_preset_list)
        layout.addWidget(self._form_row(("Preset name", self.preset_name), ("", save_button)))
        layout.addWidget(self._form_row(("Available", self.preset_combo), ("", load_button), ("", refresh_button)))

    def _build_output_controls(self) -> None:
        self.output_panel = QtWidgets.QWidget()
        output_layout = QtWidgets.QVBoxLayout(self.output_panel)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(10)

        action_grid = QtWidgets.QGridLayout()
        update_button = QtWidgets.QPushButton("Update Visualizations")
        export_button = QtWidgets.QPushButton("Export Coordinates")
        save_2d_button = QtWidgets.QPushButton("Save 2D Screenshot")
        show_3d_button = QtWidgets.QPushButton("Show 3D Window")
        show_2d_button = QtWidgets.QPushButton("Show 2D Window")
        plot_vertical_grid_button = QtWidgets.QPushButton("Plot Vertical Grid")
        save_grid_levels_button = QtWidgets.QPushButton("Save grid levels")
        update_button.clicked.connect(self.update_visualization)
        export_button.clicked.connect(self.export_coordinates)
        save_2d_button.clicked.connect(self.save_2d_screenshot)
        show_3d_button.clicked.connect(self.show_3d)
        show_2d_button.clicked.connect(self.two_d_window.show)
        plot_vertical_grid_button.clicked.connect(self.plot_vertical_grid)
        save_grid_levels_button.clicked.connect(self.save_grid_levels)
        action_grid.addWidget(update_button, 0, 0)
        action_grid.addWidget(export_button, 0, 1)
        action_grid.addWidget(save_2d_button, 0, 2)
        action_grid.addWidget(show_3d_button, 1, 0)
        action_grid.addWidget(show_2d_button, 1, 1)
        action_grid.addWidget(plot_vertical_grid_button, 1, 2)
        action_grid.addWidget(save_grid_levels_button, 2, 0)
        for column in range(3):
            action_grid.setColumnStretch(column, 1)
        output_layout.addLayout(action_grid)

        self.warning_box = QtWidgets.QPlainTextEdit()
        self.warning_box.setReadOnly(True)
        self.warning_box.setPlaceholderText("Validation warnings appear here.")
        self.warning_box.setMaximumBlockCount(100)
        self.warning_box.setFixedHeight(80)
        self.summary_box = QtWidgets.QTextBrowser()
        self.summary_box.setOpenExternalLinks(False)
        self.summary_box.setMinimumHeight(360)
        output_layout.addWidget(QtWidgets.QLabel("Domain and Grid Summary"))
        output_layout.addWidget(self.summary_box)
        output_layout.addWidget(QtWidgets.QLabel("Validation Warnings"))
        output_layout.addWidget(self.warning_box)

        self.output_scroll = QtWidgets.QScrollArea()
        self.output_scroll.setWidgetResizable(True)
        self.output_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.output_scroll.setMinimumHeight(180)
        self.output_scroll.setWidget(self.output_panel)
        self.panel_splitter.addWidget(self.output_scroll)

    def _cluster_editor(self, index: int, values: dict | None = None) -> dict:
        values = values or {}
        editor = {
            "nx_turbines": self._int(values.get("nx_turbines", 4), minimum=1),
            "ny_turbines": self._int(values.get("ny_turbines", 4), minimum=1),
            "spacing_x_D": self._float(values.get("spacing_x_D", 7.0), minimum=0.001, step=0.5),
            "spacing_y_D": self._float(values.get("spacing_y_D", 5.0), minimum=0.001, step=0.5),
            "staggered": QtWidgets.QCheckBox("Staggered rows"),
            "center_in_x": QtWidgets.QCheckBox("Center cluster in x"),
            "center_in_y": QtWidgets.QCheckBox("Center cluster in y"),
            "x_start": self._float(values.get("x_start", 1000.0 + 2000.0 * (index - 1)), step=100.0),
            "y_center": self._float(values.get("y_center", 3000.0), step=100.0),
            "rotation_deg": self._float(values.get("rotation_deg", 0.0), step=5.0),
        }
        editor["staggered"].setChecked(values.get("staggered", False))
        editor["center_in_x"].setChecked(values.get("center_in_x", False))
        editor["center_in_y"].setChecked(values.get("center_in_y", False))
        editor["center_in_x"].toggled.connect(lambda enabled, state=editor: state["x_start"].setDisabled(enabled))
        editor["center_in_y"].toggled.connect(lambda enabled, state=editor: state["y_center"].setDisabled(enabled))
        editor["x_start"].setDisabled(editor["center_in_x"].isChecked())
        editor["y_center"].setDisabled(editor["center_in_y"].isChecked())
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.addWidget(self._form_row(("N_x turbines", editor["nx_turbines"]), ("N_y turbines", editor["ny_turbines"]), ("", editor["staggered"])))
        layout.addWidget(self._form_row(("S_x / D", editor["spacing_x_D"]), ("S_y / D", editor["spacing_y_D"])))
        layout.addWidget(self._form_row(("", editor["center_in_x"]), ("", editor["center_in_y"])))
        layout.addWidget(self._form_row(("x_start (m)", editor["x_start"]), ("y_center (m)", editor["y_center"]), ("Rotation (deg)", editor["rotation_deg"])))
        editor["page"] = page
        return editor

    def _rebuild_cluster_editors(self, count: int) -> None:
        prior = [self._editor_values(editor) for editor in self.cluster_editors]
        self.cluster_editors = []
        self._clear_toolbox(self.cluster_toolbox)
        for index in range(1, count + 1):
            editor = self._cluster_editor(index, prior[index - 1] if index <= len(prior) else None)
            self.cluster_editors.append(editor)
            self.cluster_toolbox.addItem(editor["page"], f"Cluster {index}")

    def _amr_editor(self, index: int, values: dict | None = None) -> dict:
        values = values or self._default_amr_values(index)
        editor = {
            "ratio_x": self._float(values.get("ratio_x", 2.0), minimum=0.001, step=0.5),
            "ratio_y": self._float(values.get("ratio_y", 2.0), minimum=0.001, step=0.5),
            "ratio_z": self._float(values.get("ratio_z", 2.0), minimum=0.001, step=0.5),
            "x_lo": self._float(values.get("x_lo", 0.0), step=10.0),
            "x_hi": self._float(values.get("x_hi", self.lx.value()), step=10.0),
            "y_lo": self._float(values.get("y_lo", 0.0), step=10.0),
            "y_hi": self._float(values.get("y_hi", self.ly.value()), step=10.0),
            "z_lo": self._float(values.get("z_lo", 0.0), step=10.0),
            "z_hi": self._float(values.get("z_hi", self.lz.value()), step=10.0),
            "color": QtWidgets.QComboBox(),
        }
        editor["color"].addItems(AMR_COLORS)
        editor["color"].setCurrentText(values.get("color", AMR_COLORS[(index - 1) % len(AMR_COLORS)]))
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.addWidget(self._form_row(("r_x", editor["ratio_x"]), ("r_y", editor["ratio_y"]), ("r_z", editor["ratio_z"]), ("Box color", editor["color"])))
        layout.addWidget(self._form_row(("x_lo", editor["x_lo"]), ("x_hi", editor["x_hi"]), ("y_lo", editor["y_lo"]), ("y_hi", editor["y_hi"])))
        layout.addWidget(self._form_row(("z_lo", editor["z_lo"]), ("z_hi", editor["z_hi"])))
        editor["page"] = page
        return editor

    def _default_amr_values(self, index: int) -> dict:
        if self.amr_editors:
            parent = self._editor_values(self.amr_editors[-1])
        else:
            parent = {"x_lo": -self.lin.value() if self.inflow_enabled.isChecked() else 0.0, "x_hi": self.lx.value(), "y_lo": 0.0, "y_hi": self.ly.value(), "z_lo": 0.0, "z_hi": self.lz.value()}
        result = {"ratio_x": 2.0, "ratio_y": 2.0, "ratio_z": 2.0, "color": AMR_COLORS[(index - 1) % len(AMR_COLORS)]}
        for axis in ("x", "y", "z"):
            lo, hi = parent[f"{axis}_lo"], parent[f"{axis}_hi"]
            margin = 0.2 * (hi - lo)
            result[f"{axis}_lo"] = lo + margin
            result[f"{axis}_hi"] = hi - margin
        return result

    def _rebuild_amr_editors(self, count: int) -> None:
        prior = [self._editor_values(editor) for editor in self.amr_editors]
        self.amr_editors = []
        self._clear_toolbox(self.amr_toolbox)
        for index in range(1, count + 1):
            editor = self._amr_editor(index, prior[index - 1] if index <= len(prior) else None)
            self.amr_editors.append(editor)
            self.amr_toolbox.addItem(editor["page"], f"Level {index}")

    @staticmethod
    def _editor_values(editor: dict) -> dict:
        values = {}
        for key, widget in editor.items():
            if key == "page":
                continue
            if isinstance(widget, QtWidgets.QCheckBox):
                values[key] = widget.isChecked()
            elif isinstance(widget, QtWidgets.QComboBox):
                values[key] = widget.currentText()
            elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                values[key] = widget.value()
        return values

    @staticmethod
    def _clear_toolbox(toolbox: QtWidgets.QToolBox) -> None:
        while toolbox.count():
            page = toolbox.widget(0)
            toolbox.removeItem(0)
            page.deleteLater()

    def _build_domain_config(self) -> DomainConfig:
        return DomainConfig(
            Lx=self.lx.value(), Ly=self.ly.value(), Lz=self.lz.value(),
            grid_mode=self.grid_mode.currentData(),
            dx=self.dx.value(), dy=self.dy.value(), dz=self.dz.value(),
            Nx=self.nx.value(), Ny=self.ny.value(), Nz=self.nz.value(),
            vertical_discretization=self.vertical_discretization.currentData(),
            initial_dz=self.initial_dz.value(),
            stretching_ratio=(
                self.stretching_ratio.value()
                if self.stretched_parameter_mode.currentData() == "ratio" else None
            ),
            stretched_parameter_mode=self.stretched_parameter_mode.currentData(),
            stretched_final_dz=(
                self.stretched_final_dz.value()
                if self.stretched_parameter_mode.currentData() == "final_dz" else None
            ),
            tanh_initial_dz=self.tanh_initial_dz.value(),
            tanh_stretching_factor=(
                self.tanh_stretching_factor.value()
                if self.tanh_parameter_mode.currentData() == "factor" else None
            ),
            tanh_parameter_mode=self.tanh_parameter_mode.currentData(),
            tanh_final_dz=(
                self.tanh_final_dz.value()
                if self.tanh_parameter_mode.currentData() == "final_dz" else None
            ),
            piecewise_uniform_dz=self.piecewise_uniform_dz.value(),
            piecewise_transition_height=self.piecewise_transition_height.value(),
            piecewise_parameter_mode=self.piecewise_parameter_mode.currentData(),
            piecewise_stretching_ratio=(
                self.piecewise_stretching_ratio.value()
                if self.piecewise_parameter_mode.currentData() == "ratio" else None
            ),
            piecewise_final_dz=(
                self.piecewise_final_dz.value()
                if self.piecewise_parameter_mode.currentData() == "final_dz" else None
            ),
            inflow_enabled=self.inflow_enabled.isChecked(), Lin=self.lin.value(),
            rayleigh_enabled=self.rayleigh_enabled.isChecked(), rayleigh_depth=self.rayleigh_depth.value(),
            max_level=self.max_level.value(),
            amr_levels=[AMRLevelConfig(level_index=index, **self._editor_values(editor)) for index, editor in enumerate(self.amr_editors, start=1)],
        )

    def _build_study_config(self) -> FarmStudyConfig:
        clusters = [ClusterConfig(cluster_id=index, **self._editor_values(editor)) for index, editor in enumerate(self.cluster_editors, start=1)]
        return FarmStudyConfig(self.rotor_diameter.value(), self.hub_height.value(), clusters)

    def _compute_state(self) -> tuple[DomainConfig, FarmStudyConfig, dict, dict, dict, list[str]]:
        domain = self._build_domain_config()
        study = self._build_study_config()
        metrics = compute_grid_metrics(domain)
        amr = compute_amr_metrics(domain, study.rotor_diameter, study.hub_height)
        layouts = generate_all_layouts(study, domain_config=domain)
        warnings = validate_configuration(domain, study, layouts) + amr["warnings"]
        return domain, study, metrics, amr, layouts, warnings

    def _summary_html(
        self,
        domain: DomainConfig,
        study: FarmStudyConfig,
        metrics: dict,
        amr: dict,
        layouts: dict,
    ) -> str:
        counts, spacing, lengths = metrics["counts"], metrics["spacing"], metrics["domain_lengths"]
        vertical = metrics["vertical_grid"]
        total_counts = amr["total_counts"] if amr["levels"] else counts
        total_cells = amr["total_cells"] if amr["levels"] else metrics["total_cells"]
        total_cube = amr["cube_equivalent"] if amr["levels"] else metrics["cube_equivalent"]
        rotor_bottom = max(0.0, study.hub_height - 0.5 * study.rotor_diameter)
        vertical_profile = compute_vertical_grid_profile(domain)
        farm_vertical_cells = min(
            counts["Nz"],
            max(0, int(np.searchsorted(vertical_profile["z_levels"], rotor_bottom, side="right") - 1)),
        )
        rows = [
            ("Main L_x (m)", lengths["Lx_main"]), ("Solved L_x (m)", lengths["Lx_total"]), ("L_y (m)", lengths["Ly"]), ("L_z (m)", lengths["Lz"]),
            ("Vertical discretization", vertical["type"].replace("_", " ").title()),
            ("Domain vertical cells (N_z)", counts["Nz"]),
            ("Farm vertical cells to rotor bottom", farm_vertical_cells),
            ("Base N_x / N_y / N_z", f"{counts['Nx']} / {counts['Ny']} / {counts['Nz']}"),
            ("Total N_x / N_y / N_z", f"{total_counts['Nx']} / {total_counts['Ny']} / {total_counts['Nz']}"),
            ("Base d_x / d_y (m)", f"{spacing['dx']:.3f} / {spacing['dy']:.3f}"),
            ("Total cells", f"{total_cells:,}"), ("Cube-equivalent", f"~{total_cube:.3f}^3"), ("Total turbines", layouts["total_turbines"]),
        ]
        if vertical["type"] == "uniform":
            rows.append(("d_z (m)", f"{vertical['actual_initial_dz']:.6f}"))
        elif vertical["type"] == "stretched":
            rows.append(("Requested initial d_z (m)", f"{vertical['requested_initial_dz']:.6f}"))
            if vertical["requested_final_dz"] is not None:
                rows.append(("Requested final d_z (m)", f"{vertical['requested_final_dz']:.6f}"))
            rows.extend(
                [
                    ("Actual initial d_z (m)", f"{vertical['actual_initial_dz']:.6f}"),
                    ("Actual final d_z (m)", f"{vertical['final_dz']:.6f}"),
                    ("Stretching ratio", f"{vertical['stretching_ratio']:.6f}"),
                    ("Nearest H for requested spacing", f"{vertical['nearest_height']:.6f}"),
                ]
            )
        elif vertical["type"] == "piecewise":
            rows.extend(
                [
                    ("Requested uniform d_z (m)", f"{vertical['requested_initial_dz']:.6f}"),
                    ("Requested z_t (m)", f"{vertical['requested_transition_height']:.6f}"),
                ]
            )
            if vertical["requested_final_dz"] is not None:
                rows.append(("Requested final d_z (m)", f"{vertical['requested_final_dz']:.6f}"))
            rows.extend(
                [
                    ("Actual uniform d_z (m)", f"{vertical['actual_initial_dz']:.6f}"),
                    ("Actual z_t (m)", f"{vertical['actual_transition_height']:.6f}"),
                    ("Actual final d_z (m)", f"{vertical['final_dz']:.6f}"),
                    ("Stretching ratio", f"{vertical['stretching_ratio']:.6f}"),
                    ("Uniform / stretched cells", f"{vertical['uniform_cell_count']} / {vertical['stretched_cell_count']}"),
                ]
            )
        else:
            if vertical["requested_initial_dz"] is not None:
                rows.append(("Requested initial d_z (m)", f"{vertical['requested_initial_dz']:.6f}"))
            if vertical["requested_final_dz"] is not None:
                rows.append(("Requested final d_z (m)", f"{vertical['requested_final_dz']:.6f}"))
            rows.extend(
                [
                    ("Actual initial d_z (m)", f"{vertical['actual_initial_dz']:.6f}"),
                    ("Final d_z (m)", f"{vertical['final_dz']:.6f}"),
                    ("Tanh stretching factor", f"{vertical['tanh_stretching_factor']:.6f}"),
                ]
            )
        def table(title: str, pairs: list[tuple[str, object]]) -> str:
            body = "".join(f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>" for key, value in pairs)
            return f"<h3>{title}</h3><table border='1' cellspacing='0' cellpadding='4'>{body}</table>"
        blocks = [table("Domain and Grid Summary", rows)]
        if amr["levels"]:
            level_blocks = []
            for level in amr["levels"]:
                level_rows = [
                    ("Color", level["color"]), ("Ratios", f"{level['ratios']['x']:.3f}, {level['ratios']['y']:.3f}, {level['ratios']['z']:.3f}"),
                    ("Bounds", f"x: {level['bounds']['x_lo']:.1f}-{level['bounds']['x_hi']:.1f}; y: {level['bounds']['y_lo']:.1f}-{level['bounds']['y_hi']:.1f}; z: {level['bounds']['z_lo']:.1f}-{level['bounds']['z_hi']:.1f}"),
                    ("N_x / N_y / N_z", f"{level['counts']['Nx']} / {level['counts']['Ny']} / {level['counts']['Nz']}"),
                    ("d_x / d_y / d_z (m)", f"{level['spacing']['dx']:.3f} / {level['spacing']['dy']:.3f} / {level['spacing']['dz']:.3f}"),
                    ("Total cells", f"{level['total_cells']:,}"), ("z points to rotor", f"{level['z_points_to_rotor']:.3f}"),
                ]
                level_blocks.append(f"<div style='display:inline-block;vertical-align:top;margin-right:18px'>{table(f'AMR Level {level['level_index']}', level_rows)}</div>")
            blocks.append("<h3>AMR Level Summaries</h3>" + "".join(level_blocks))
        return "".join(blocks)

    def update_visualization(self) -> None:
        try:
            domain, study, metrics, amr, layouts, warnings = self._compute_state()
            self.warning_box.setPlainText("\n".join(warnings) if warnings else "No validation warnings.")
            self.summary_box.setHtml(self._summary_html(domain, study, metrics, amr, layouts))
            self._update_3d(domain, study, metrics, amr, layouts)
            self.two_d_window.update_figures(domain, study, layouts, amr)
            self.show_3d()
            self.two_d_window.show()
        except Exception as error:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Update failed", str(error))

    def _update_3d(self, domain: DomainConfig, study: FarmStudyConfig, metrics: dict, amr: dict, layouts: dict) -> None:
        created_plotter = self.plotter is None
        if self.plotter is None:
            self.plotter = BackgroundPlotter(show=False, app=QtWidgets.QApplication.instance(), title="FarmVis 3D Domain")
            self._configure_camera_toolbar()

        cluster_values = tuple(
            tuple(getattr(cluster, field_name) for field_name in ClusterConfig.__dataclass_fields__)
            for cluster in study.clusters
        )
        layout_coordinates = tuple(
            (cluster["coordinates"].shape, cluster["coordinates"].tobytes())
            for cluster in layouts["clusters"]
        )
        rotor_angle_signature = (
            study.rotor_diameter,
            study.hub_height,
            cluster_values,
        )
        turbine_signature = (
            rotor_angle_signature,
            layout_coordinates,
        )
        if rotor_angle_signature != self._rotor_angle_signature:
            rng = np.random.default_rng()
            self._rotor_angles = [
                rng.uniform(0.0, 360.0, size=cluster["turbine_count"])
                for cluster in layouts["clusters"]
            ]
            self._rotor_angle_signature = rotor_angle_signature

        turbines_changed = created_plotter or turbine_signature != self._turbine_signature
        if turbines_changed:
            self._turbine_signature = turbine_signature

        populate_3d_scene(
            self.plotter,
            domain,
            study,
            layouts,
            amr_metrics=amr,
            metrics=metrics,
            rotor_angles=self._rotor_angles,
            update_turbines=turbines_changed,
            reset_camera=created_plotter,
        )

    def _configure_camera_toolbar(self) -> None:
        if self.plotter is None:
            return

        camera_views = (
            ("Top (-Z)", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
            ("Bottom (+Z)", (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
            ("Front / Upstream (+X)", (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            ("Back / Downstream (-X)", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            ("Left (+Y)", (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
            ("Right (-Y)", (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            ("Isometric", (-1.0, -1.0, 1.0), (0.0, 0.0, 1.0)),
        )
        actions = self.plotter.default_camera_tool_bar.actions()
        for action, (label, direction, view_up) in zip(actions, camera_views, strict=False):
            action.setText(label)
            action.triggered.disconnect()
            action.triggered.connect(
                lambda _checked=False, vector=direction, up=view_up: self.plotter.view_vector(vector, up)
            )

    def show_3d(self) -> None:
        if self.plotter is not None:
            # BackgroundPlotter is the render widget; its Qt MainWindow owns visibility.
            window = self.plotter.app_window
            window.showNormal()
            window.raise_()
            window.activateWindow()
            self.plotter.render()

    def plot_vertical_grid(self) -> None:
        try:
            domain = self._build_domain_config()
            if self.vertical_grid_window is None:
                self.vertical_grid_window = VerticalGridWindow()
            self.vertical_grid_window.update_profile(domain)
            self.vertical_grid_window.showNormal()
            self.vertical_grid_window.raise_()
            self.vertical_grid_window.activateWindow()
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "Vertical grid plot failed", str(error))

    def save_grid_levels(self) -> None:
        output, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Vertical Grid Levels",
            str(OUTPUT_DIR / "vertical_grid_levels.txt"),
            "Text file (*.txt)",
        )
        if not output:
            return
        output_path = Path(output)
        if output_path.suffix.lower() != ".txt":
            output_path = output_path.with_suffix(".txt")
        try:
            saved = export_vertical_grid_levels(self._build_domain_config(), output_path)
            QtWidgets.QMessageBox.information(self, "Grid levels saved", f"Grid levels saved to:\n{saved}")
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "Grid level export failed", str(error))

    def save_2d_screenshot(self) -> None:
        output, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save 2D Screenshot",
            str(OUTPUT_DIR / "farmvis_2d.png"),
            "PNG image (*.png)",
        )
        if not output:
            return

        output_path = Path(output)
        if output_path.suffix.lower() != ".png":
            output_path = output_path.with_suffix(".png")

        self.two_d_window.show()
        self.two_d_window.raise_()
        self.two_d_window.activateWindow()
        QtWidgets.QApplication.processEvents()
        canvas = self.two_d_window.tabs.currentWidget()
        if canvas is None or not canvas.grab().save(str(output_path), "PNG"):
            QtWidgets.QMessageBox.critical(self, "Save failed", f"Could not save the screenshot to:\n{output_path}")
            return
        QtWidgets.QMessageBox.information(self, "Screenshot saved", f"2D screenshot saved to:\n{output_path}")

    def export_coordinates(self) -> None:
        try:
            domain, _, _, _, layouts, _ = self._compute_state()
            offset = domain.Lin if domain.inflow_enabled else 0.0
            output = export_cluster_coordinates(layouts, OUTPUT_DIR / "turbine_coordinates.txt", x_offset=offset)
            QtWidgets.QMessageBox.information(self, "Export complete", f"Coordinates saved to:\n{output}")
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(error))

    def _refresh_preset_list(self) -> None:
        selected = self.preset_combo.currentData()
        self.preset_combo.clear()
        presets = list_layout_presets(PRESET_DIR)
        for path in presets:
            self.preset_combo.addItem(path.stem, path)
        if selected is not None:
            index = self.preset_combo.findData(selected)
            if index >= 0:
                self.preset_combo.setCurrentIndex(index)

    def save_preset(self) -> None:
        name = "".join(char for char in self.preset_name.text().strip() if char not in '<>:"/\\|?*')
        if not name:
            QtWidgets.QMessageBox.warning(self, "Invalid name", "Enter a preset name containing at least one valid character.")
            return
        try:
            output = save_layout_preset(self._build_domain_config(), self._build_study_config(), PRESET_DIR / f"{name}.json")
            self._refresh_preset_list()
            index = self.preset_combo.findData(output)
            if index >= 0:
                self.preset_combo.setCurrentIndex(index)
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(error))

    def load_preset(self) -> None:
        path = self.preset_combo.currentData()
        if path is None:
            QtWidgets.QMessageBox.warning(self, "No preset", "Choose a preset first.")
            return
        try:
            domain, study = configs_from_preset(load_layout_preset(path))
            self._apply_configs(domain, study)
            self.preset_name.setText(Path(path).stem)
            self.update_visualization()
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(error))

    def _apply_configs(self, domain: DomainConfig, study: FarmStudyConfig) -> None:
        self.lx.setValue(domain.Lx); self.ly.setValue(domain.Ly); self.lz.setValue(domain.Lz)
        self.grid_mode.setCurrentIndex(0 if domain.grid_mode == "resolution" else 1)
        vertical_index = self.vertical_discretization.findData(domain.vertical_discretization)
        self.vertical_discretization.setCurrentIndex(max(0, vertical_index))
        if domain.dx is not None: self.dx.setValue(domain.dx)
        if domain.dy is not None: self.dy.setValue(domain.dy)
        if domain.dz is not None: self.dz.setValue(domain.dz)
        if domain.Nx is not None: self.nx.setValue(domain.Nx)
        if domain.Ny is not None: self.ny.setValue(domain.Ny)
        if domain.Nz is not None: self.nz.setValue(domain.Nz)
        if domain.initial_dz is not None: self.initial_dz.setValue(domain.initial_dz)
        if domain.stretching_ratio is not None: self.stretching_ratio.setValue(domain.stretching_ratio)
        stretched_mode_index = self.stretched_parameter_mode.findData(domain.stretched_parameter_mode)
        self.stretched_parameter_mode.setCurrentIndex(max(0, stretched_mode_index))
        if domain.stretched_final_dz is not None:
            self.stretched_final_dz.setValue(domain.stretched_final_dz)
        if domain.tanh_initial_dz is not None: self.tanh_initial_dz.setValue(domain.tanh_initial_dz)
        if domain.tanh_stretching_factor is not None:
            self.tanh_stretching_factor.setValue(domain.tanh_stretching_factor)
        tanh_mode_index = self.tanh_parameter_mode.findData(domain.tanh_parameter_mode)
        self.tanh_parameter_mode.setCurrentIndex(max(0, tanh_mode_index))
        if domain.tanh_final_dz is not None: self.tanh_final_dz.setValue(domain.tanh_final_dz)
        if domain.piecewise_uniform_dz is not None:
            self.piecewise_uniform_dz.setValue(domain.piecewise_uniform_dz)
        if domain.piecewise_transition_height is not None:
            self.piecewise_transition_height.setValue(domain.piecewise_transition_height)
        piecewise_mode_index = self.piecewise_parameter_mode.findData(domain.piecewise_parameter_mode)
        self.piecewise_parameter_mode.setCurrentIndex(max(0, piecewise_mode_index))
        if domain.piecewise_stretching_ratio is not None:
            self.piecewise_stretching_ratio.setValue(domain.piecewise_stretching_ratio)
        if domain.piecewise_final_dz is not None:
            self.piecewise_final_dz.setValue(domain.piecewise_final_dz)
        self.inflow_enabled.setChecked(domain.inflow_enabled); self.lin.setValue(domain.Lin)
        self.rayleigh_enabled.setChecked(domain.rayleigh_enabled); self.rayleigh_depth.setValue(domain.rayleigh_depth)
        self.max_level.setValue(domain.max_level if domain.vertical_discretization == "uniform" else 0)
        self.rotor_diameter.setValue(study.rotor_diameter)
        self.hub_height.setValue(study.hub_height)
        self._sync_turbine_preset(study.rotor_diameter, study.hub_height)
        self.cluster_count.setValue(len(study.clusters))
        for editor, cluster in zip(self.cluster_editors, study.clusters):
            for key in ClusterConfig.__dataclass_fields__:
                if key == "cluster_id": continue
                value = getattr(cluster, key)
                widget = editor[key]
                if isinstance(widget, QtWidgets.QCheckBox): widget.setChecked(value)
                else: widget.setValue(value)
        for editor, level in zip(self.amr_editors, domain.amr_levels):
            for key in AMRLevelConfig.__dataclass_fields__:
                if key == "level_index": continue
                value = getattr(level, key)
                widget = editor[key]
                if isinstance(widget, QtWidgets.QComboBox): widget.setCurrentText(value)
                else: widget.setValue(value)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.plotter is not None:
            self.plotter.close()
        self.two_d_window.close()
        if self.vertical_grid_window is not None:
            self.vertical_grid_window.close()
        event.accept()


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = FarmVisWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
