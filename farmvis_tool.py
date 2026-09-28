from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.patches import Circle, Rectangle


@dataclass(slots=True)
class DomainConfig:
    Lx: float
    Ly: float
    Lz: float
    grid_mode: str
    dx: float | None = None
    dy: float | None = None
    dz: float | None = None
    Nx: int | None = None
    Ny: int | None = None
    Nz: int | None = None
    inflow_enabled: bool = False
    Lin: float = 0.0
    rayleigh_enabled: bool = False
    rayleigh_depth: float = 0.0
    max_level: int = 0
    amr_levels: list["AMRLevelConfig"] = field(default_factory=list)


@dataclass(slots=True)
class AMRLevelConfig:
    level_index: int
    ratio_x: float
    ratio_y: float
    ratio_z: float
    x_lo: float
    x_hi: float
    y_lo: float
    y_hi: float
    z_lo: float
    z_hi: float
    color: str = "red"


@dataclass(slots=True)
class ClusterConfig:
    cluster_id: int
    nx_turbines: int
    ny_turbines: int
    spacing_x_D: float
    spacing_y_D: float
    staggered: bool
    x_start: float
    y_center: float
    rotation_deg: float = 0.0
    center_in_x: bool = False
    center_in_y: bool = False


@dataclass(slots=True)
class FarmStudyConfig:
    rotor_diameter: float
    hub_height: float
    clusters: list[ClusterConfig]


def compute_grid_metrics(domain_config: DomainConfig) -> dict[str, Any]:
    _ensure_positive(domain_config.Lx, "Lx")
    _ensure_positive(domain_config.Ly, "Ly")
    _ensure_positive(domain_config.Lz, "Lz")

    inflow_length = domain_config.Lin if domain_config.inflow_enabled else 0.0
    if domain_config.inflow_enabled:
        _ensure_nonnegative(domain_config.Lin, "Lin")

    if domain_config.rayleigh_enabled:
        _ensure_nonnegative(domain_config.rayleigh_depth, "rayleigh_depth")

    total_lengths = {
        "x": domain_config.Lx + inflow_length,
        "y": domain_config.Ly,
        "z": domain_config.Lz,
    }

    grid_mode = domain_config.grid_mode.strip().lower()
    if grid_mode == "resolution":
        dx = _require_value(domain_config.dx, "dx")
        dy = _require_value(domain_config.dy, "dy")
        dz = _require_value(domain_config.dz, "dz")
        _ensure_positive(dx, "dx")
        _ensure_positive(dy, "dy")
        _ensure_positive(dz, "dz")

        counts = {
            "x": max(1, int(round(total_lengths["x"] / dx))),
            "y": max(1, int(round(total_lengths["y"] / dy))),
            "z": max(1, int(round(total_lengths["z"] / dz))),
        }
    elif grid_mode == "counts":
        counts = {
            "x": int(_require_value(domain_config.Nx, "Nx")),
            "y": int(_require_value(domain_config.Ny, "Ny")),
            "z": int(_require_value(domain_config.Nz, "Nz")),
        }
        _ensure_positive(counts["x"], "Nx")
        _ensure_positive(counts["y"], "Ny")
        _ensure_positive(counts["z"], "Nz")
    else:
        raise ValueError("grid_mode must be either 'resolution' or 'counts'")

    spacing = {
        axis: total_lengths[axis] / counts[axis]
        for axis in ("x", "y", "z")
    }
    total_cells = counts["x"] * counts["y"] * counts["z"]

    return {
        "grid_mode": grid_mode,
        "domain_bounds": {
            "main": (0.0, domain_config.Lx, 0.0, domain_config.Ly, 0.0, domain_config.Lz),
            "full": (-inflow_length, domain_config.Lx, 0.0, domain_config.Ly, 0.0, domain_config.Lz),
            "inflow": (-inflow_length, 0.0, 0.0, domain_config.Ly, 0.0, domain_config.Lz)
            if domain_config.inflow_enabled and inflow_length > 0.0
            else None,
            "rayleigh": (
                -inflow_length,
                domain_config.Lx,
                0.0,
                domain_config.Ly,
                max(0.0, domain_config.Lz - domain_config.rayleigh_depth),
                domain_config.Lz,
            )
            if domain_config.rayleigh_enabled and domain_config.rayleigh_depth > 0.0
            else None,
        },
        "domain_lengths": {
            "Lx_main": domain_config.Lx,
            "Lx_total": total_lengths["x"],
            "Ly": total_lengths["y"],
            "Lz": total_lengths["z"],
            "Lin": inflow_length,
            "rayleigh_depth": domain_config.rayleigh_depth if domain_config.rayleigh_enabled else 0.0,
        },
        "counts": {"Nx": counts["x"], "Ny": counts["y"], "Nz": counts["z"]},
        "spacing": {"dx": spacing["x"], "dy": spacing["y"], "dz": spacing["z"]},
        "total_cells": total_cells,
        "cube_equivalent": total_cells ** (1.0 / 3.0),
    }


def compute_amr_metrics(
    domain_config: DomainConfig,
    rotor_diameter: float,
    hub_height: float,
) -> dict[str, Any]:
    base_metrics = compute_grid_metrics(domain_config)
    warnings: list[str] = []
    _ensure_nonnegative(domain_config.max_level, "max_level")
    _ensure_positive(rotor_diameter, "rotor_diameter")
    _ensure_positive(hub_height, "hub_height")

    levels: list[dict[str, Any]] = []
    total_counts = base_metrics["counts"].copy()
    total_cells = base_metrics["total_cells"]
    if domain_config.max_level == 0:
        return {
            "max_level": 0,
            "levels": [],
            "warnings": warnings,
            "total_counts": total_counts,
            "total_cells": total_cells,
            "cube_equivalent": total_cells ** (1.0 / 3.0),
        }

    if len(domain_config.amr_levels) != domain_config.max_level:
        warnings.append(
            f"Maximum level is {domain_config.max_level}, but {len(domain_config.amr_levels)} AMR levels were provided."
        )

    previous_spacing = base_metrics["spacing"].copy()
    parent_bounds = base_metrics["domain_bounds"]["full"]
    rotor_bottom = hub_height - 0.5 * rotor_diameter

    for level_config in domain_config.amr_levels[: domain_config.max_level]:
        ratios = {
            "x": level_config.ratio_x,
            "y": level_config.ratio_y,
            "z": level_config.ratio_z,
        }
        for axis, value in ratios.items():
            if value <= 0:
                raise ValueError(f"AMR Level {level_config.level_index} ratio_{axis} must be positive")
            if value <= 1.0:
                warnings.append(
                    f"AMR Level {level_config.level_index} ratio_{axis}={value:.3f} is not greater than 1 and does not refine that direction."
                )

        bounds = (
            float(level_config.x_lo),
            float(level_config.x_hi),
            float(level_config.y_lo),
            float(level_config.y_hi),
            float(level_config.z_lo),
            float(level_config.z_hi),
        )
        lengths = {
            "x": bounds[1] - bounds[0],
            "y": bounds[3] - bounds[2],
            "z": bounds[5] - bounds[4],
        }
        for axis, value in lengths.items():
            if value <= 0:
                warnings.append(
                    f"AMR Level {level_config.level_index} has invalid {axis}_lo/{axis}_hi bounds; hi must exceed lo."
                )

        if not _bounds_contained(bounds, parent_bounds):
            parent_label = "the solved domain" if level_config.level_index == 1 else f"AMR Level {level_config.level_index - 1}"
            warnings.append(
                f"AMR Level {level_config.level_index} is not fully contained inside {parent_label}."
            )

        level_spacing = {
            axis: previous_spacing[f"d{axis}"] / ratios[axis]
            for axis in ("x", "y", "z")
        }
        level_counts = {
            axis: max(1, int(round(lengths[axis] / level_spacing[axis])))
            for axis in ("x", "y", "z")
        }
        parent_overlap_counts = {
            axis: max(1, int(round(lengths[axis] / previous_spacing[f"d{axis}"])))
            for axis in ("x", "y", "z")
        }
        effective_lengths = {
            axis: level_counts[axis] * level_spacing[axis]
            for axis in ("x", "y", "z")
        }
        for axis in ("x", "y", "z"):
            mismatch = abs(effective_lengths[axis] - lengths[axis])
            if mismatch > max(1e-9, 1e-6 * max(lengths[axis], 1.0)):
                warnings.append(
                    f"AMR Level {level_config.level_index} {axis}-length does not divide evenly by refined spacing; "
                    f"rounded to N_{axis}={level_counts[axis]} with effective length {effective_lengths[axis]:.6f} m."
                )

        level_total_cells = level_counts["x"] * level_counts["y"] * level_counts["z"]
        parent_overlap_total_cells = (
            parent_overlap_counts["x"] * parent_overlap_counts["y"] * parent_overlap_counts["z"]
        )
        z_points_to_rotor = rotor_bottom / level_spacing["z"]
        level_metrics = {
            "level_index": level_config.level_index,
            "color": level_config.color,
            "ratios": ratios,
            "bounds": {
                "x_lo": bounds[0],
                "x_hi": bounds[1],
                "y_lo": bounds[2],
                "y_hi": bounds[3],
                "z_lo": bounds[4],
                "z_hi": bounds[5],
            },
            "box_lengths": {
                "Lx": lengths["x"],
                "Ly": lengths["y"],
                "Lz": lengths["z"],
            },
            "spacing": {
                "dx": level_spacing["x"],
                "dy": level_spacing["y"],
                "dz": level_spacing["z"],
            },
            "counts": {
                "Nx": level_counts["x"],
                "Ny": level_counts["y"],
                "Nz": level_counts["z"],
            },
            "parent_overlap_counts": {
                "Nx": parent_overlap_counts["x"],
                "Ny": parent_overlap_counts["y"],
                "Nz": parent_overlap_counts["z"],
            },
            "effective_lengths": {
                "Lx": effective_lengths["x"],
                "Ly": effective_lengths["y"],
                "Lz": effective_lengths["z"],
            },
            "total_cells": level_total_cells,
            "parent_overlap_total_cells": parent_overlap_total_cells,
            "cube_equivalent": level_total_cells ** (1.0 / 3.0),
            "rotor_points": {
                "x": rotor_diameter / level_spacing["x"],
                "y": rotor_diameter / level_spacing["y"],
                "z": rotor_diameter / level_spacing["z"],
            },
            "z_points_to_rotor": z_points_to_rotor,
        }
        levels.append(level_metrics)
        total_counts["Nx"] += level_metrics["counts"]["Nx"] - level_metrics["parent_overlap_counts"]["Nx"]
        total_counts["Ny"] += level_metrics["counts"]["Ny"] - level_metrics["parent_overlap_counts"]["Ny"]
        total_counts["Nz"] += level_metrics["counts"]["Nz"] - level_metrics["parent_overlap_counts"]["Nz"]
        total_cells += level_metrics["total_cells"] - level_metrics["parent_overlap_total_cells"]
        previous_spacing = level_metrics["spacing"]
        parent_bounds = bounds

    return {
        "max_level": domain_config.max_level,
        "levels": levels,
        "warnings": warnings,
        "total_counts": total_counts,
        "total_cells": total_cells,
        "cube_equivalent": total_cells ** (1.0 / 3.0),
    }


def generate_cluster_layout(
    cluster_config: ClusterConfig,
    rotor_diameter: float,
    domain_config: DomainConfig | None = None,
) -> dict[str, Any]:
    _ensure_positive(rotor_diameter, "rotor_diameter")
    _ensure_positive(cluster_config.nx_turbines, "nx_turbines")
    _ensure_positive(cluster_config.ny_turbines, "ny_turbines")
    _ensure_positive(cluster_config.spacing_x_D, "spacing_x_D")
    _ensure_positive(cluster_config.spacing_y_D, "spacing_y_D")

    spacing_x = cluster_config.spacing_x_D * rotor_diameter
    spacing_y = cluster_config.spacing_y_D * rotor_diameter
    x_span = (cluster_config.nx_turbines - 1) * spacing_x

    coordinates: list[list[float]] = []
    for i in range(cluster_config.nx_turbines):
        x_value = cluster_config.x_start + i * spacing_x
        stagger_shift = 0.5 * spacing_y if cluster_config.staggered and i % 2 == 1 else 0.0
        for j in range(cluster_config.ny_turbines):
            y_value = (
                cluster_config.y_center
                + (j - (cluster_config.ny_turbines - 1) / 2.0) * spacing_y
                + stagger_shift
            )
            coordinates.append([x_value, y_value])

    coords = np.asarray(coordinates, dtype=float)

    if cluster_config.staggered:
        # Recenter the staggered layout so y_center matches the array centroid.
        coords[:, 1] -= float(coords[:, 1].mean()) - cluster_config.y_center

    current_centroid = coords.mean(axis=0)
    target_centroid = current_centroid.copy()

    if cluster_config.center_in_x:
        if domain_config is None:
            raise ValueError("domain_config must be provided when center_in_x is enabled")
        target_centroid[0] = 0.5 * domain_config.Lx

    if cluster_config.center_in_y:
        if domain_config is None:
            raise ValueError("domain_config must be provided when center_in_y is enabled")
        target_centroid[1] = 0.5 * domain_config.Ly

    if cluster_config.center_in_x or cluster_config.center_in_y:
        coords += target_centroid - current_centroid

    center = coords.mean(axis=0)

    if abs(cluster_config.rotation_deg) > 0.0:
        theta = np.deg2rad(cluster_config.rotation_deg)
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=float,
        )
        coords = (coords - center) @ rotation.T + center

    if not cluster_config.center_in_x:
        translation_x = cluster_config.x_start - float(coords[:, 0].min())
        coords[:, 0] += translation_x
        center[0] += translation_x

    return {
        "cluster_id": cluster_config.cluster_id,
        "coordinates": coords,
        "center": center,
        "turbine_count": int(coords.shape[0]),
        "x_span": x_span,
        "y_span": (cluster_config.ny_turbines - 1) * spacing_y,
    }


def generate_all_layouts(
    study_config: FarmStudyConfig,
    domain_config: DomainConfig | None = None,
) -> dict[str, Any]:
    _ensure_positive(study_config.rotor_diameter, "rotor_diameter")
    _ensure_positive(study_config.hub_height, "hub_height")

    layouts = [
        generate_cluster_layout(cluster, study_config.rotor_diameter, domain_config=domain_config)
        for cluster in study_config.clusters
    ]
    total_turbines = sum(layout["turbine_count"] for layout in layouts)

    return {
        "rotor_diameter": study_config.rotor_diameter,
        "hub_height": study_config.hub_height,
        "clusters": layouts,
        "total_turbines": total_turbines,
    }


def serialize_configs(
    domain_config: DomainConfig,
    study_config: FarmStudyConfig,
    schema_version: int = 1,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "domain": {
            "Lx": domain_config.Lx,
            "Ly": domain_config.Ly,
            "Lz": domain_config.Lz,
            "grid_mode": domain_config.grid_mode,
            "dx": domain_config.dx,
            "dy": domain_config.dy,
            "dz": domain_config.dz,
            "Nx": domain_config.Nx,
            "Ny": domain_config.Ny,
            "Nz": domain_config.Nz,
            "inflow_enabled": domain_config.inflow_enabled,
            "Lin": domain_config.Lin,
            "rayleigh_enabled": domain_config.rayleigh_enabled,
            "rayleigh_depth": domain_config.rayleigh_depth,
            "max_level": domain_config.max_level,
            "amr_levels": [
                {
                    "level_index": level.level_index,
                    "ratio_x": level.ratio_x,
                    "ratio_y": level.ratio_y,
                    "ratio_z": level.ratio_z,
                    "x_lo": level.x_lo,
                    "x_hi": level.x_hi,
                    "y_lo": level.y_lo,
                    "y_hi": level.y_hi,
                    "z_lo": level.z_lo,
                    "z_hi": level.z_hi,
                    "color": level.color,
                }
                for level in domain_config.amr_levels
            ],
        },
        "study": {
            "rotor_diameter": study_config.rotor_diameter,
            "hub_height": study_config.hub_height,
            "clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "nx_turbines": cluster.nx_turbines,
                    "ny_turbines": cluster.ny_turbines,
                    "spacing_x_D": cluster.spacing_x_D,
                    "spacing_y_D": cluster.spacing_y_D,
                    "staggered": cluster.staggered,
                    "x_start": cluster.x_start,
                    "y_center": cluster.y_center,
                    "center_in_x": cluster.center_in_x,
                    "center_in_y": cluster.center_in_y,
                    "rotation_deg": cluster.rotation_deg,
                }
                for cluster in study_config.clusters
            ],
        },
    }


def save_layout_preset(
    domain_config: DomainConfig,
    study_config: FarmStudyConfig,
    path: str | Path,
    schema_version: int = 1,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_configs(domain_config, study_config, schema_version=schema_version)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def load_layout_preset(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    validate_preset_payload(payload)
    return payload


def list_layout_presets(directory: str | Path) -> list[Path]:
    preset_dir = Path(directory)
    if not preset_dir.exists():
        return []
    return sorted(preset_dir.glob("*.json"))


def configs_from_preset(payload: dict[str, Any]) -> tuple[DomainConfig, FarmStudyConfig]:
    validate_preset_payload(payload)

    domain_data = payload["domain"]
    study_data = payload["study"]
    clusters = [
        ClusterConfig(
            cluster_id=int(cluster_data["cluster_id"]),
            nx_turbines=int(cluster_data["nx_turbines"]),
            ny_turbines=int(cluster_data["ny_turbines"]),
            spacing_x_D=float(cluster_data["spacing_x_D"]),
            spacing_y_D=float(cluster_data["spacing_y_D"]),
            staggered=bool(cluster_data["staggered"]),
            x_start=float(cluster_data["x_start"]),
            y_center=float(cluster_data["y_center"]),
            center_in_x=bool(cluster_data.get("center_in_x", False)),
            center_in_y=bool(cluster_data.get("center_in_y", False)),
            rotation_deg=float(cluster_data.get("rotation_deg", 0.0)),
        )
        for cluster_data in study_data["clusters"]
    ]

    domain_config = DomainConfig(
        Lx=float(domain_data["Lx"]),
        Ly=float(domain_data["Ly"]),
        Lz=float(domain_data["Lz"]),
        grid_mode=str(domain_data["grid_mode"]),
        dx=_optional_float(domain_data.get("dx")),
        dy=_optional_float(domain_data.get("dy")),
        dz=_optional_float(domain_data.get("dz")),
        Nx=_optional_int(domain_data.get("Nx")),
        Ny=_optional_int(domain_data.get("Ny")),
        Nz=_optional_int(domain_data.get("Nz")),
        inflow_enabled=bool(domain_data.get("inflow_enabled", False)),
        Lin=float(domain_data.get("Lin", 0.0)),
        rayleigh_enabled=bool(domain_data.get("rayleigh_enabled", False)),
        rayleigh_depth=float(domain_data.get("rayleigh_depth", 0.0)),
        max_level=int(domain_data.get("max_level", 0)),
        amr_levels=[
            AMRLevelConfig(
                level_index=int(level_data["level_index"]),
                ratio_x=float(level_data["ratio_x"]),
                ratio_y=float(level_data["ratio_y"]),
                ratio_z=float(level_data["ratio_z"]),
                x_lo=float(level_data["x_lo"]),
                x_hi=float(level_data["x_hi"]),
                y_lo=float(level_data["y_lo"]),
                y_hi=float(level_data["y_hi"]),
                z_lo=float(level_data["z_lo"]),
                z_hi=float(level_data["z_hi"]),
                color=str(level_data.get("color", "red")),
            )
            for level_data in domain_data.get("amr_levels", [])
        ],
    )
    study_config = FarmStudyConfig(
        rotor_diameter=float(study_data["rotor_diameter"]),
        hub_height=float(study_data["hub_height"]),
        clusters=clusters,
    )
    return domain_config, study_config


def export_cluster_coordinates(
    layouts: dict[str, Any],
    path: str | Path,
    x_offset: float = 0.0,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for cluster_index, cluster in enumerate(layouts["clusters"]):
        for x_coord, y_coord in cluster["coordinates"]:
            lines.append(f"{x_coord + x_offset:.6f} {y_coord:.6f}")
        if cluster_index < len(layouts["clusters"]) - 1:
            lines.append("")

    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def validate_configuration(
    domain_config: DomainConfig,
    study_config: FarmStudyConfig,
    layouts: dict[str, Any] | None = None,
) -> list[str]:
    warnings: list[str] = []

    if domain_config.rayleigh_enabled and domain_config.rayleigh_depth > domain_config.Lz:
        warnings.append("Rayleigh damping depth exceeds L_z and extends below the domain floor.")

    if domain_config.inflow_enabled and domain_config.Lin <= 0.0:
        warnings.append("Inflow region is enabled but L_in is not positive.")

    if layouts is None:
        layouts = generate_all_layouts(study_config, domain_config=domain_config)

    rotor_radius = 0.5 * study_config.rotor_diameter
    rotor_top = study_config.hub_height + rotor_radius
    rotor_bottom = study_config.hub_height - rotor_radius

    if rotor_top > domain_config.Lz:
        warnings.append(
            f"Rotor top at z={rotor_top:.3f} m exceeds the domain height L_z={domain_config.Lz:.3f} m."
        )
    if rotor_bottom < 0.0:
        warnings.append(
            f"Rotor bottom at z={rotor_bottom:.3f} m lies below the domain floor z=0.000 m."
        )

    rayleigh_floor = domain_config.Lz - domain_config.rayleigh_depth
    if domain_config.rayleigh_enabled and domain_config.rayleigh_depth > 0.0 and rotor_top > rayleigh_floor:
        warnings.append(
            f"Rotor top at z={rotor_top:.3f} m intersects the Rayleigh damping region starting at z={rayleigh_floor:.3f} m."
        )

    for cluster in layouts["clusters"]:
        cluster_id = cluster["cluster_id"]
        coords = cluster["coordinates"]
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]

        if np.any(x_coords < 0.0) or np.any(x_coords > domain_config.Lx):
            warnings.append(
                f"Cluster {cluster_id} has turbine centers outside the main-domain x-bounds [0, {domain_config.Lx:.3f}] m."
            )
        if np.any(y_coords < 0.0) or np.any(y_coords > domain_config.Ly):
            warnings.append(
                f"Cluster {cluster_id} has turbine centers outside the domain y-bounds [0, {domain_config.Ly:.3f}] m."
            )
        if not domain_config.inflow_enabled and np.any(x_coords < 0.0):
            warnings.append(
                f"Cluster {cluster_id} extends into negative x while the inflow region is disabled."
            )

    return warnings


def validate_preset_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Preset payload must be a JSON object.")

    if "domain" not in payload or "study" not in payload:
        raise ValueError("Preset payload must contain 'domain' and 'study' sections.")

    domain_data = payload["domain"]
    study_data = payload["study"]
    required_domain = {"Lx", "Ly", "Lz", "grid_mode"}
    required_study = {"rotor_diameter", "hub_height", "clusters"}

    missing_domain = sorted(required_domain - set(domain_data))
    if missing_domain:
        raise ValueError(f"Preset domain section is missing required keys: {', '.join(missing_domain)}")

    missing_study = sorted(required_study - set(study_data))
    if missing_study:
        raise ValueError(f"Preset study section is missing required keys: {', '.join(missing_study)}")

    if not isinstance(study_data["clusters"], list) or len(study_data["clusters"]) == 0:
        raise ValueError("Preset study.clusters must be a non-empty list.")

    if "amr_levels" in domain_data:
        if not isinstance(domain_data["amr_levels"], list):
            raise ValueError("Preset domain.amr_levels must be a list when provided.")
        required_amr = {
            "level_index",
            "ratio_x",
            "ratio_y",
            "ratio_z",
            "x_lo",
            "x_hi",
            "y_lo",
            "y_hi",
            "z_lo",
            "z_hi",
            "color",
        }
        for level_index, level_data in enumerate(domain_data["amr_levels"], start=1):
            missing_amr = sorted(required_amr - set(level_data))
            if missing_amr:
                raise ValueError(
                    f"Preset AMR level {level_index} is missing required keys: {', '.join(missing_amr)}"
                )

    required_cluster = {
        "cluster_id",
        "nx_turbines",
        "ny_turbines",
        "spacing_x_D",
        "spacing_y_D",
        "staggered",
        "x_start",
        "y_center",
    }
    for cluster_index, cluster_data in enumerate(study_data["clusters"], start=1):
        missing_cluster = sorted(required_cluster - set(cluster_data))
        if missing_cluster:
            raise ValueError(
                f"Preset cluster {cluster_index} is missing required keys: {', '.join(missing_cluster)}"
            )


def build_3d_scene(
    domain_config: DomainConfig,
    study_config: FarmStudyConfig,
    layouts: dict[str, Any],
    amr_metrics: dict[str, Any] | None = None,
) -> pv.Plotter:
    metrics = compute_grid_metrics(domain_config)
    pv.set_jupyter_backend("client")

    plotter = pv.Plotter(notebook=True)
    populate_3d_scene(plotter, domain_config, study_config, layouts, amr_metrics=amr_metrics, metrics=metrics)
    return plotter


def populate_3d_scene(
    plotter: pv.Plotter,
    domain_config: DomainConfig,
    study_config: FarmStudyConfig,
    layouts: dict[str, Any],
    amr_metrics: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    rotor_angles: list[np.ndarray] | None = None,
    update_turbines: bool = True,
    reset_camera: bool = True,
) -> None:
    if metrics is None:
        metrics = compute_grid_metrics(domain_config)

    plotter.set_background("white")
    plotter.add_mesh(
        _box_line_polydata(metrics["domain_bounds"]["main"]),
        color="black",
        line_width=2,
        name="main-domain",
    )

    inflow_bounds = metrics["domain_bounds"]["inflow"]
    if inflow_bounds is not None:
        plotter.add_mesh(
            _dashed_box_polydata(inflow_bounds),
            color="black",
            line_width=2,
            name="inflow-region",
        )
    else:
        plotter.remove_actor("inflow-region", reset_camera=False, render=False)

    rayleigh_bounds = metrics["domain_bounds"]["rayleigh"]
    if rayleigh_bounds is not None:
        plotter.add_mesh(
            _rayleigh_surface_polydata(rayleigh_bounds),
            color="lightgray",
            opacity=0.15,
            show_edges=False,
            name="rayleigh-region",
        )
    else:
        plotter.remove_actor("rayleigh-region", reset_camera=False, render=False)

    for actor_name in tuple(plotter.actors):
        if actor_name.startswith("amr-level-"):
            plotter.remove_actor(actor_name, reset_camera=False, render=False)
    if amr_metrics is not None:
        for level in amr_metrics["levels"]:
            plotter.add_mesh(
                _box_line_polydata(_bounds_dict_to_tuple(level["bounds"])),
                color=level["color"],
                line_width=2,
                name=f"amr-level-{level['level_index']}",
            )

    if update_turbines:
        turbines = _turbine_line_polydata(
            layouts,
            rotor_diameter=study_config.rotor_diameter,
            hub_height=study_config.hub_height,
            rotor_angles=rotor_angles,
        )
        if turbines.n_cells:
            plotter.add_mesh(turbines, color="black", line_width=2, name="turbines")
        else:
            plotter.remove_actor("turbines", reset_camera=False, render=False)

    if reset_camera:
        plotter.view_isometric()


def plot_xy(
    domain_config: DomainConfig,
    study_config: FarmStudyConfig,
    layouts: dict[str, Any],
    amr_metrics: dict[str, Any] | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    _add_xy_domain(ax, domain_config)
    if amr_metrics is not None:
        _add_xy_amr(ax, amr_metrics)

    half_diameter = 0.5 * study_config.rotor_diameter
    for cluster in layouts["clusters"]:
        coords = cluster["coordinates"]
        for x_coord, y_coord in coords:
            ax.plot(
                [x_coord, x_coord],
                [y_coord - half_diameter, y_coord + half_diameter],
                color="black",
                linewidth=1.3,
            )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-domain_config.Lin if domain_config.inflow_enabled else 0.0, domain_config.Lx)
    ax.set_ylim(0.0, domain_config.Ly)
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    return fig


def plot_xz(
    domain_config: DomainConfig,
    study_config: FarmStudyConfig,
    layouts: dict[str, Any],
    amr_metrics: dict[str, Any] | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    _add_xz_domain(ax, domain_config)
    if amr_metrics is not None:
        _add_xz_amr(ax, amr_metrics)

    rotor_radius = 0.5 * study_config.rotor_diameter
    for cluster in layouts["clusters"]:
        x_coords = cluster["coordinates"][:, 0]
        for x_coord in x_coords:
            ax.plot(
                [x_coord, x_coord],
                [study_config.hub_height - rotor_radius, study_config.hub_height + rotor_radius],
                color="black",
                linewidth=1.3,
            )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-domain_config.Lin if domain_config.inflow_enabled else 0.0, domain_config.Lx)
    ax.set_ylim(0.0, domain_config.Lz)
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    return fig


def plot_yz(
    domain_config: DomainConfig,
    study_config: FarmStudyConfig,
    layouts: dict[str, Any],
    amr_metrics: dict[str, Any] | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    _add_yz_domain(ax, domain_config)
    if amr_metrics is not None:
        _add_yz_amr(ax, amr_metrics)

    rotor_radius = 0.5 * study_config.rotor_diameter
    for cluster in layouts["clusters"]:
        y_coords = cluster["coordinates"][:, 1]
        for y_coord in y_coords:
            ax.add_patch(
                Circle(
                    (y_coord, study_config.hub_height),
                    rotor_radius,
                    fill=False,
                    color="black",
                    linewidth=1.3,
                )
            )

    ax.set_xlabel("y (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, domain_config.Ly)
    ax.set_ylim(0.0, domain_config.Lz)
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    return fig


def _box_line_polydata(bounds: tuple[float, float, float, float, float, float]) -> pv.PolyData:
    return _merge_line_segments([pv.Line(start, end) for start, end in _box_edges(bounds)])


def _turbine_line_polydata(
    layouts: dict[str, Any],
    rotor_diameter: float,
    hub_height: float,
    rotor_angles: list[np.ndarray] | None = None,
) -> pv.PolyData:
    rotor_radius = 0.5 * rotor_diameter
    segments: list[pv.PolyData] = []
    rng = np.random.default_rng()

    if rotor_angles is not None and len(rotor_angles) != len(layouts["clusters"]):
        raise ValueError("rotor_angles must contain one angle array per cluster")

    for cluster_index, cluster in enumerate(layouts["clusters"]):
        cluster_angles = (
            np.asarray(rotor_angles[cluster_index], dtype=float)
            if rotor_angles is not None
            else rng.uniform(0.0, 360.0, size=cluster["turbine_count"])
        )
        if len(cluster_angles) != cluster["turbine_count"]:
            raise ValueError("Each rotor angle array must contain one angle per turbine")

        for (x_coord, y_coord), rotor_angle in zip(cluster["coordinates"], cluster_angles, strict=True):
            hub = np.array([float(x_coord), float(y_coord), hub_height], dtype=float)
            ground = np.array([float(x_coord), float(y_coord), 0.0], dtype=float)
            segments.append(pv.Line(ground, hub))

            for blade_offset in (0.0, 120.0, 240.0):
                angle_deg = rotor_angle + blade_offset
                angle = np.deg2rad(angle_deg)
                blade_tip = hub + np.array(
                    [0.0, rotor_radius * np.cos(angle), rotor_radius * np.sin(angle)],
                    dtype=float,
                )
                segments.append(pv.Line(hub, blade_tip))

    return _merge_line_segments(segments) if segments else pv.PolyData()


def _rayleigh_surface_polydata(
    bounds: tuple[float, float, float, float, float, float],
) -> pv.PolyData:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    points = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=float,
    )
    # Four exterior side strips only; omit both horizontal faces.
    faces = np.array(
        [
            4, 0, 1, 5, 4,
            4, 1, 2, 6, 5,
            4, 2, 3, 7, 6,
            4, 3, 0, 4, 7,
        ],
        dtype=np.int64,
    )
    return pv.PolyData(points, faces)


def _dashed_box_polydata(
    bounds: tuple[float, float, float, float, float, float],
    dash_fraction: float = 0.06,
    gap_fraction: float = 0.04,
) -> pv.PolyData:
    segments = []
    for start, end in _box_edges(bounds):
        edge_length = float(np.linalg.norm(end - start))
        dash_length = max(edge_length * dash_fraction, edge_length / 80.0)
        gap_length = max(edge_length * gap_fraction, edge_length / 120.0)
        distance = 0.0
        direction = (end - start) / edge_length
        while distance < edge_length:
            dash_end = min(distance + dash_length, edge_length)
            p0 = start + direction * distance
            p1 = start + direction * dash_end
            segments.append(pv.Line(p0, p1))
            distance = dash_end + gap_length

    return _merge_line_segments(segments)


def _merge_line_segments(segments: list[pv.PolyData]) -> pv.PolyData:
    merged = segments[0]
    for segment in segments[1:]:
        merged = merged.merge(segment)
    return merged


def _box_edges(bounds: tuple[float, float, float, float, float, float]) -> list[tuple[np.ndarray, np.ndarray]]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    corners = {
        "000": np.array([xmin, ymin, zmin], dtype=float),
        "001": np.array([xmin, ymin, zmax], dtype=float),
        "010": np.array([xmin, ymax, zmin], dtype=float),
        "011": np.array([xmin, ymax, zmax], dtype=float),
        "100": np.array([xmax, ymin, zmin], dtype=float),
        "101": np.array([xmax, ymin, zmax], dtype=float),
        "110": np.array([xmax, ymax, zmin], dtype=float),
        "111": np.array([xmax, ymax, zmax], dtype=float),
    }
    edge_keys = [
        ("000", "100"),
        ("010", "110"),
        ("001", "101"),
        ("011", "111"),
        ("000", "010"),
        ("100", "110"),
        ("001", "011"),
        ("101", "111"),
        ("000", "001"),
        ("100", "101"),
        ("010", "011"),
        ("110", "111"),
    ]
    return [(corners[a], corners[b]) for a, b in edge_keys]


def _add_xy_domain(ax: plt.Axes, domain_config: DomainConfig) -> None:
    ax.add_patch(Rectangle((0.0, 0.0), domain_config.Lx, domain_config.Ly, fill=False, color="black", linewidth=1.5))
    if domain_config.inflow_enabled and domain_config.Lin > 0.0:
        ax.add_patch(
            Rectangle(
                (-domain_config.Lin, 0.0),
                domain_config.Lin,
                domain_config.Ly,
                fill=False,
                color="black",
                linewidth=1.2,
                linestyle="--",
            )
        )


def _add_xz_domain(ax: plt.Axes, domain_config: DomainConfig) -> None:
    ax.add_patch(Rectangle((0.0, 0.0), domain_config.Lx, domain_config.Lz, fill=False, color="black", linewidth=1.5))
    if domain_config.inflow_enabled and domain_config.Lin > 0.0:
        ax.add_patch(
            Rectangle(
                (-domain_config.Lin, 0.0),
                domain_config.Lin,
                domain_config.Lz,
                fill=False,
                color="black",
                linewidth=1.2,
                linestyle="--",
            )
        )
    if domain_config.rayleigh_enabled and domain_config.rayleigh_depth > 0.0:
        xmin = -domain_config.Lin if domain_config.inflow_enabled else 0.0
        zmin = max(0.0, domain_config.Lz - domain_config.rayleigh_depth)
        ax.plot(
            [domain_config.Lx, domain_config.Lx, np.nan, xmin, xmin],
            [domain_config.Lz, zmin, np.nan, domain_config.Lz, zmin],
            color="lightgray",
            linewidth=2.0,
            alpha=0.5,
        )


def _add_yz_domain(ax: plt.Axes, domain_config: DomainConfig) -> None:
    ax.add_patch(Rectangle((0.0, 0.0), domain_config.Ly, domain_config.Lz, fill=False, color="black", linewidth=1.5))
    if domain_config.rayleigh_enabled and domain_config.rayleigh_depth > 0.0:
        zmin = max(0.0, domain_config.Lz - domain_config.rayleigh_depth)
        ax.plot(
            [domain_config.Ly, domain_config.Ly, np.nan, 0.0, 0.0],
            [domain_config.Lz, zmin, np.nan, domain_config.Lz, zmin],
            color="lightgray",
            linewidth=2.0,
            alpha=0.5,
        )


def _add_xy_amr(ax: plt.Axes, amr_metrics: dict[str, Any]) -> None:
    for level in amr_metrics["levels"]:
        bounds = level["bounds"]
        ax.add_patch(
            Rectangle(
                (bounds["x_lo"], bounds["y_lo"]),
                bounds["x_hi"] - bounds["x_lo"],
                bounds["y_hi"] - bounds["y_lo"],
                fill=False,
                color=level["color"],
                linewidth=1.5,
            )
        )


def _add_xz_amr(ax: plt.Axes, amr_metrics: dict[str, Any]) -> None:
    for level in amr_metrics["levels"]:
        bounds = level["bounds"]
        ax.add_patch(
            Rectangle(
                (bounds["x_lo"], bounds["z_lo"]),
                bounds["x_hi"] - bounds["x_lo"],
                bounds["z_hi"] - bounds["z_lo"],
                fill=False,
                color=level["color"],
                linewidth=1.5,
            )
        )


def _add_yz_amr(ax: plt.Axes, amr_metrics: dict[str, Any]) -> None:
    for level in amr_metrics["levels"]:
        bounds = level["bounds"]
        ax.add_patch(
            Rectangle(
                (bounds["y_lo"], bounds["z_lo"]),
                bounds["y_hi"] - bounds["y_lo"],
                bounds["z_hi"] - bounds["z_lo"],
                fill=False,
                color=level["color"],
                linewidth=1.5,
            )
        )


def _bounds_dict_to_tuple(bounds: dict[str, float]) -> tuple[float, float, float, float, float, float]:
    return (
        bounds["x_lo"],
        bounds["x_hi"],
        bounds["y_lo"],
        bounds["y_hi"],
        bounds["z_lo"],
        bounds["z_hi"],
    )


def _bounds_contained(
    inner_bounds: tuple[float, float, float, float, float, float],
    outer_bounds: tuple[float, float, float, float, float, float],
) -> bool:
    return (
        inner_bounds[0] >= outer_bounds[0]
        and inner_bounds[1] <= outer_bounds[1]
        and inner_bounds[2] >= outer_bounds[2]
        and inner_bounds[3] <= outer_bounds[3]
        and inner_bounds[4] >= outer_bounds[4]
        and inner_bounds[5] <= outer_bounds[5]
    )


def _require_value(value: float | int | None, name: str) -> float | int:
    if value is None:
        raise ValueError(f"{name} must be provided")
    return value


def _ensure_positive(value: float | int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _ensure_nonnegative(value: float | int, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


__all__ = [
    "AMRLevelConfig",
    "ClusterConfig",
    "DomainConfig",
    "FarmStudyConfig",
    "build_3d_scene",
    "compute_amr_metrics",
    "configs_from_preset",
    "compute_grid_metrics",
    "export_cluster_coordinates",
    "generate_all_layouts",
    "generate_cluster_layout",
    "list_layout_presets",
    "load_layout_preset",
    "plot_xy",
    "plot_xz",
    "plot_yz",
    "populate_3d_scene",
    "save_layout_preset",
    "serialize_configs",
    "validate_configuration",
    "validate_preset_payload",
]
