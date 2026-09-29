# WindFarmVis

WindFarmVis is a desktop visualization and configuration tool for constructing wind-farm domains for ERF large-eddy simulations (LES). It provides a native Qt control panel, an interactive PyVista 3D view, and Matplotlib-based 2D projections.

The application is intended for planning domain dimensions, numerical resolution, turbine layouts, inflow regions, Rayleigh damping regions, and nested adaptive mesh refinement (AMR) boxes before preparing an ERF input case.

## Features

- Configure the main LES domain dimensions and base-grid resolution.
- Add an optional upstream turbulence-inflow region.
- Add an optional Rayleigh damping region at the top of the domain.
- Define one or more wind-farm clusters.
- Configure turbine count, spacing, staggering, placement, centering, and rotation independently for each cluster.
- Define nested AMR levels with directional refinement ratios and custom box extents.
- Visualize the domain, turbines, inflow region, damping region, and AMR boxes in an interactive 3D window.
- View `x-y`, `x-z`, and `y-z` projections in a separate tabbed window.
- Review base-grid and AMR grid statistics, rotor resolution, turbine counts, and validation warnings.
- Save and reload complete input configurations as JSON presets.
- Export turbine coordinates as an ERF-compatible two-column text file.

## Application Windows

FarmVis opens three native windows:

1. **Control panel** — organizes inputs into **Domain**, **Wind Farm**, **AMR**, and **Presets** tabs, with validation messages, summaries, and actions available below every tab.
2. **3D domain view** — an interactive PyVista window for rotating, panning, and zooming around the domain.
3. **2D views** — a tabbed window containing the `x-y`, `x-z`, and `y-z` projections.

The visualization windows are created when the control panel starts and are updated in place. Closing or minimizing a visualization window does not discard the current inputs. Use **Show 3D Window** or **Show 2D Window** to bring it back.

Drag the horizontal divider below the control tabs to make the tab area taller or shorter. The shared action buttons, domain and grid summary, and compact validation-warning pane remain available below the divider.

## Repository Layout

```text
farmvis/
├── farmvis_gui.py       # Native Qt application and GUI state management
├── farmvis_tool.py      # Domain, grid, layout, validation, export, and plotting logic
├── environment.yml      # Conda environment definition
├── presets/             # Saved JSON layout configurations
├── outputs/             # Generated turbine-coordinate files
└── README.md
```

The application creates `presets/` and `outputs/` automatically if they do not already exist.

## Requirements

- Windows, Linux, or macOS with desktop OpenGL support
- Miniconda, Anaconda, Miniforge, or another Conda-compatible installation
- Python 3.14, as specified by the supplied environment
- A graphical desktop session for Qt and VTK rendering

The principal runtime packages are:

- NumPy
- Matplotlib
- PyVista and VTK
- PyVistaQt
- PySide6

All dependencies are installed from `conda-forge` by the supplied `environment.yml` file.

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/ZaneFrey/WindFarmVis.git
cd farmvis
```

Replace the example URL with the actual repository URL.

### 2. Create the Conda environment

```powershell
conda env create -f environment.yml
```

This creates an environment named `farmvis`.

### 3. Activate the environment

```powershell
conda activate farmvis
```

### 4. Launch FarmVis

```powershell
python farmvis_gui.py
```

If `conda` is not available in the current PowerShell session, launch an Anaconda/Miniforge prompt or initialize Conda for PowerShell first:

```powershell
conda init powershell
```

Restart PowerShell after running that command.

## Updating an Existing Environment

After pulling repository changes that modify `environment.yml`, update the environment with:

```powershell
conda env update -n farmvis -f environment.yml --prune
```

The `--prune` option removes packages that are no longer declared by the environment definition.

## Basic Workflow

1. Launch `farmvis_gui.py` from the activated `farmvis` environment.
2. Set the domain dimensions and base-grid resolution in the **Domain** tab.
3. Set the turbine model dimensions and define one or more clusters in the **Wind Farm** tab.
4. Configure any nested refinement boxes in the **AMR** tab.
5. Click **Update Visualizations**.
6. Inspect validation warnings and numerical summaries in the control panel.
7. Rotate and inspect the native 3D view, then review the three 2D projections.
8. Save the configuration as a JSON preset from the **Presets** tab if it will be reused.
9. Click **Export Coordinates** to generate the turbine-coordinate input file.

Changing a control does not immediately rebuild the visualization. Click **Update Visualizations** after making changes.

## Domain Controls

### Domain dimensions

- **L_x (m)** — streamwise length of the main domain, excluding the optional inflow region.
- **L_y (m)** — spanwise domain width.
- **L_z (m)** — vertical domain height.

The main domain is represented by:

```text
x = [0, L_x]
y = [0, L_y]
z = [0, L_z]
```

### Grid mode

FarmVis supports two base-grid input modes.

#### Resolution input

Enter target values for `d_x`, `d_y`, and `d_z`. FarmVis rounds the corresponding cell counts to the nearest positive integers and reports the effective spacing implied by those counts.

#### Grid counts

Enter `N_x`, `N_y`, and `N_z` directly. FarmVis calculates spacing as:

```text
d_x = total solved x length / N_x
d_y = L_y / N_y
d_z = L_z / N_z
```

`N_x`, `N_y`, and `N_z` represent cell counts rather than node counts.

### Inflow region

Enable **Enable inflow region** and enter `L_in` to prepend an upstream turbulence-generation region.

The visual coordinate range becomes:

```text
x = [-L_in, L_x]
```

The solved streamwise length used in the base-grid calculation is `L_in + L_x`. The inflow region is drawn as a dashed black box.

### Rayleigh damping region

Enable **Enable Rayleigh region** and enter a depth to reserve a damping layer at the top of the domain.

The damping region occupies the full solved-domain footprint, including the inflow region when enabled:

```text
x = [-L_in, L_x]
y = [0, L_y]
z = [L_z - depth, L_z]
```

It remains inside `L_z`; it does not increase the domain height. The visualization consists only of faint, semitransparent strips on the four exterior side faces, with no top face, interior lower face, or filled volume. The vertical 2D projections show the corresponding side boundary segments.

## AMR Controls

Set **Maximum level** to `0` to use only the base grid. Increasing it creates one expandable control page per refinement level.

Each level includes:

- **r_x, r_y, r_z** — directional refinement ratios relative to the immediately preceding level.
- **x_lo/x_hi** — absolute streamwise bounds of the refinement box.
- **y_lo/y_hi** — absolute spanwise bounds of the refinement box.
- **z_lo/z_hi** — absolute vertical bounds of the refinement box.
- **Box color** — outline color used in the 3D and 2D visualizations.

Refined spacing is calculated recursively:

```text
d_x(level) = d_x(previous level) / r_x
d_y(level) = d_y(previous level) / r_y
d_z(level) = d_z(previous level) / r_z
```

Level 1 must lie inside the full solved domain. Each subsequent level should lie completely inside its parent level. FarmVis reports warnings for boxes outside their parent bounds.

AMR coordinates use the visual coordinate system. Therefore, negative `x` bounds are valid only when an inflow region exists and only down to `-L_in`.

### AMR cell accounting

FarmVis reports local cell counts for each refinement box. The top-level **Total cells** value is overlap-aware:

1. Start with all Level 0 cells.
2. Remove parent-level cells covered by Level 1.
3. Add the Level 1 cells in that region.
4. Repeat the replacement process for every deeper level.

Parent and child cells occupying the same physical volume are therefore not counted twice.

The displayed total `N_x`, `N_y`, and `N_z` values use the same replacement principle independently in each direction. For partially refined three-dimensional domains, **Total cells** is the authoritative computational-size estimate; the directional totals are useful summaries but do not define a single uniform tensor-product grid.

## Wind Farm Controls

### Shared turbine dimensions

- **Rotor D (m)** — rotor diameter shared by all clusters.
- **Hub height (m)** — rotor-center height shared by all clusters.
- **# clusters** — number of independently configured wind-farm clusters.
- **Preset** — applies rotor diameter and hub height for the NREL 5 MW, DTU 10 MW, IEA 10 MW, IEA 15 MW, Vestas v236, or Siemens Gamesa SG 11.0-200 DD turbine. The IEA 15 MW turbine is selected by default.

Changing the number of clusters creates or removes expandable cluster pages. Values in retained cluster pages are preserved.

### Per-cluster controls

- **N_x turbines** — number of streamwise turbine rows.
- **N_y turbines** — number of spanwise turbine columns.
- **S_x / D** — streamwise spacing in rotor diameters.
- **S_y / D** — spanwise spacing in rotor diameters.
- **Staggered rows** — offsets every other streamwise row by half the spanwise spacing.
- **Center cluster in x** — places the cluster centroid at `L_x / 2` and disables `x_start`.
- **Center cluster in y** — places the cluster centroid at `L_y / 2` and disables `y_center`.
- **x_start (m)** — upstream-most turbine-center coordinate after rotation when x-centering is disabled.
- **y_center (m)** — requested spanwise centroid when y-centering is disabled.
- **Rotation (deg)** — rotates the full cluster in the horizontal plane about its centroid.

Staggered layouts are recentered before rotation so that staggering does not bias the requested spanwise centroid. Cluster centering is also applied before rotation, preserving the selected centroid.

When **Center cluster in x** is disabled, the rotated array is translated so its minimum turbine `x` coordinate remains equal to `x_start`.

## Visualizations

### Interactive 3D view

The 3D window displays:

- Main-domain edges as solid black lines
- Inflow-region edges as dashed black lines
- Rayleigh region as a faint open surface shell spanning the main and inflow regions
- AMR boxes as solid colored outlines
- Turbine towers and three-bladed rotors as lightweight black line geometry, with blades spaced 120 degrees apart in the `y-z` rotor plane
- Randomized rotor orientations that remain unchanged across visualization updates until the corresponding turbine configuration changes

Typical PyVista mouse controls are:

- Left drag — rotate
- Middle drag — pan
- Mouse wheel or right drag — zoom

The 3D camera uses atmospheric coordinates: `+x` is downstream, `+y` is left when facing downstream, and `+z` is upward. The initial elevated view is from upstream and to the right, looking downstream into the domain. The camera toolbar follows the same convention: Front/Upstream looks along `+x`, Back/Downstream along `-x`, Left along `+y`, Right along `-y`, Top along `-z`, and Bottom along `+z`.

Click **Show 3D Window** to restore or bring the native PyVista window to the foreground.

### 2D views

The separate 2D window contains three tabs:

- **x-y** — top-down domain and turbine layout
- **x-z** — streamwise side view
- **y-z** — frontal view with rotor disks

AMR boxes appear as colored rectangular outlines in each corresponding projection. Click **Show 2D Window** to restore this window. Select the desired projection tab and click **Save 2D Screenshot** in the control panel to save only that domain plot as a PNG image, without the surrounding tabs or window controls.

## Numerical Summaries

The control panel reports:

- Main and total solved domain dimensions
- Base-grid counts and effective spacing
- AMR-adjusted directional totals
- Overlap-aware total solved cells
- Cube-equivalent grid size
- Grid points across the rotor diameter in each direction
- Vertical grid points between the ground and rotor bottom
- Total turbine count
- Per-level AMR extents, resolution, grid shape, and cell count

The `z points to rotor` metric is:

```text
(hub height - rotor diameter / 2) / d_z
```

Validation warnings identify conditions such as turbines outside the main domain, rotors intersecting the ground or damping region, invalid AMR bounds, non-nested AMR levels, and box lengths that do not divide evenly by the refined spacing.

Warnings do not prevent visualization unless an input makes the calculation impossible, such as a non-positive required dimension or refinement ratio.

## Presets

Presets store inputs only. Derived coordinates, grid metrics, warnings, and plot state are recalculated after loading.

### Save a preset

1. Enter a name under **Preset name**.
2. Click **Save Layout**.
3. FarmVis writes `presets/<name>.json`.

Saving with an existing name replaces that preset file.

### Load a preset

1. Select a file under **Available**.
2. Click **Load Layout**.
3. FarmVis rebuilds the cluster and AMR controls, restores all inputs, and updates the visualizations.

Use **Refresh** after adding or removing preset files outside the application.

Preset JSON contains a schema version and separate domain and study sections. Older presets without AMR data load with `Maximum level = 0`.

## Coordinate Export

Click **Export Coordinates** to create:

```text
outputs/turbine_coordinates.txt
```

The file contains two whitespace-separated columns:

```text
x y
```

Each cluster is separated by one blank line. Coordinates are written with six decimal places.

The visualization uses the main-domain origin at `x = 0` and displays the inflow region at negative `x`. ERF turbine inputs account for the inflow region as part of the solved domain, so FarmVis adds `L_in` to every exported turbine `x` coordinate when inflow is enabled. The exported coordinate origin is therefore the upstream edge of the complete solved domain.

## Troubleshooting

### The application cannot import PySide6 or PyVistaQt

Confirm that the correct environment is active:

```powershell
conda activate farmvis
python -c "import PySide6, pyvistaqt; print('GUI dependencies available')"
```

If the import fails, update the environment:

```powershell
conda env update -n farmvis -f environment.yml --prune
```

### The 3D window does not appear

- Click **Update Visualizations**, then **Show 3D Window**.
- Check the taskbar in case the window opened behind the control panel.
- Confirm that the session has access to a graphical desktop and OpenGL-capable driver.
- Avoid launching the program from a headless remote shell unless that session supports GUI forwarding.

### The 3D view is blank or produces OpenGL errors

Update the graphics driver and confirm that VTK can create a renderer:

```powershell
python -c "import pyvista as pv; pv.Sphere().plot()"
```

### Changes are not visible

Click **Update Visualizations** after changing controls. The application does not continuously rerender while values are being edited.

### A preset does not appear

Confirm that it is a `.json` file inside `presets/`, then click **Refresh**.

### Coordinate exports appear shifted in x

This is expected when the inflow region is enabled. The exported ERF coordinate includes the `L_in` offset; the visualization coordinate does not.

## Development

The application is intentionally split into two layers:

- `farmvis_tool.py` contains reusable calculations, validation, serialization, export, and rendering helpers.
- `farmvis_gui.py` contains Qt widgets, window management, and conversion between GUI state and configuration dataclasses.

New numerical behavior should generally be implemented and tested in `farmvis_tool.py` first. GUI code should collect inputs, invoke those functions, and display their results.

Before submitting changes, run a syntax check from the activated environment:

```powershell
python -m py_compile farmvis_gui.py farmvis_tool.py
```

Then launch the application and verify:

- Control-panel startup
- Dynamic cluster and AMR page creation
- Preset save/load round trips
- Coordinate export
- Native 3D window interaction
- All three 2D projections
- Base-grid and AMR summary calculations
