"""Mesh extraction utilities for b3_2d."""

import logging
import re
import math
import numpy as np
import pyvista as pv
from cgfoil.models import Skin, Web, Ply, Thickness

from .mesh_utils import bb_size, sort_points_by_y

logger = logging.getLogger(__name__)


def _apply_global_rotation_to_vector(vec: list[float]) -> list[float]:
    """Apply the same coordinate transform that is done to the mesh
    (rotate_z(-90) + rotate_x(180)) so web orientation ends up in the correct 2D frame."""
    ox, oy, oz = vec
    # rotate_z(-90)
    rx = oy
    ry = -ox
    rz = oz
    # rotate_x(180)
    rx2 = rx
    ry2 = -ry
    rz2 = -rz
    return [rx2, ry2, rz2]


def extract_airfoil_and_web_points(section_mesh: pv.PolyData) -> tuple:
    """Extract airfoil and web points, applying de-offset and de-twist."""
    panel_ids = section_mesh.cell_data["panel_id"]
    unique_panel_ids = np.unique(panel_ids)
    negative_panel_ids = [pid for pid in unique_panel_ids if pid < 0]

    twist = np.unique(section_mesh.cell_data["twist"])[0]
    dx = np.unique(section_mesh.cell_data["dx"])[0]
    dy = np.unique(section_mesh.cell_data["dy"])[0]
    logger.info(f"Applying translation dx={dx}, dy={dy} and twist={twist} degrees")

    section_mesh = section_mesh.rotate_z(-twist)

    if not negative_panel_ids:
        logger.warning("No negative panel_ids found, no TE or webs.")
        section = section_mesh.threshold(value=(0, panel_ids.max()), scalars="panel_id")
        airfoil = section
        points_2d = airfoil.points[:, :2].tolist()
        web_data = []
        logger.info(f"Extracted {len(points_2d)} airfoil points")
        return points_2d, web_data, airfoil

    min_panel_id = min(negative_panel_ids)
    web_panel_ids = sorted([pid for pid in negative_panel_ids if pid != min_panel_id], reverse=True)

    te = sort_points_by_y(section_mesh.threshold(value=(min_panel_id, min_panel_id), scalars="panel_id"))
    section = section_mesh.threshold(value=(0, panel_ids.max()), scalars="panel_id")
    if bb_size(te) > 0.03 * bb_size(section):
        airfoil = pv.merge([section, te])
    else:
        airfoil = section
    points_2d = airfoil.points[:, :2].tolist()
    logger.info(f"Extracted {len(points_2d)} airfoil points")

    web_data = []
    for pid in web_panel_ids:
        web = section_mesh.threshold(value=(pid, pid), scalars="panel_id")
        if web.n_cells == 0:
            continue
        web_points_2d = web.points[:, :2].tolist()
        web_data.append((web_points_2d, web))
    logger.info(f"Extracted {len(web_data)} webs with panel_ids: {web_panel_ids}")
    return points_2d, web_data, airfoil


def get_thickness_and_material_arrays(mesh: pv.PolyData) -> tuple:
    """Get thickness and material arrays from mesh."""
    thickness_keys = [k for k in mesh.cell_data if re.match(r"ply_.*_thickness", k)]
    thickness_keys.sort(key=lambda x: int(re.search(r"ply_(\d+)", x).group(1)))
    material_keys = [k.replace("_thickness", "_material") for k in thickness_keys]

    logger.info(f"Thickness keys: {thickness_keys}")
    mesh_point = mesh.cell_data_to_point_data(pass_cell_data=True)

    thicknesses = {}
    for k in thickness_keys:
        cell_thick = mesh.cell_data[k]
        unique_levels = np.sort(np.unique(np.round(cell_thick, decimals=8)))
        point_thick = mesh_point.point_data[k]
        snapped = np.array([unique_levels[np.argmin(np.abs(unique_levels - v))] for v in point_thick])
        thicknesses[k] = snapped.tolist()

    materials = {k: mesh.cell_data[k] for k in material_keys}
    return thicknesses, materials


def get_ply_thicknesses_and_materials(airfoil: pv.PolyData, web_data: list) -> tuple:
    """Get ply thicknesses and materials from airfoil and web data."""
    airfoil_thicknesses, airfoil_materials = get_thickness_and_material_arrays(airfoil)
    web_thicknesses_and_materials = [get_thickness_and_material_arrays(web_mesh) for _, web_mesh in web_data]
    web_thicknesses = [t for t, _ in web_thicknesses_and_materials]
    web_materials = [m for _, m in web_thicknesses_and_materials]
    return airfoil_thicknesses, airfoil_materials, web_thicknesses, web_materials


def define_skins_and_webs(
    airfoil_thicknesses: dict,
    airfoil_materials: dict,
    web_data: list,
    web_thicknesses: list,
    web_materials: list,
    orientations: dict = None,
    twist: float = 0.0,
) -> tuple:
    """Define skins and webs from thicknesses, materials, and points.
    Supports ribbon/optional webs that are only present in some sections.
    """
    # Skins
    skin_thickness_keys = sorted(airfoil_thicknesses, key=lambda k: int(re.search(r"ply_(\d+)", k).group(1)))
    skins = {}
    for i, key in enumerate(skin_thickness_keys, 1):
        mat_key = key.replace("_thickness", "_material")
        material = int(np.max(airfoil_materials[mat_key]))
        skins[f"skin{i}"] = Skin(
            thickness=Thickness(type="array", array=list(airfoil_thicknesses[key])),
            material=material,
            sort_index=i,
        )

    # Webs — only those present in THIS section
    web_definition = {}
    web_names_in_order = list(orientations.keys()) if orientations else [f"web{i+1}" for i in range(len(web_data))]

    for idx, (web_points_2d, _) in enumerate(web_data):
        web_name = web_names_in_order[idx] if idx < len(web_names_in_order) else f"web{idx+1}"

        thicknesses = web_thicknesses[idx]
        materials = web_materials[idx]
        ply_thickness_keys = sorted(thicknesses, key=lambda k: int(re.search(r"ply_(\d+)", k).group(1)))
        plies = []
        for key in ply_thickness_keys:
            mat_key = key.replace("_thickness", "_material")
            material = int(np.max(materials[mat_key]))
            plies.append(Ply(thickness=Thickness(type="array", array=list(thicknesses[key])), material=material))

        # Full coordinate transform for normal_ref and orientation
        if orientations and web_name in orientations:
            orig = orientations[web_name]
            rotated = _apply_global_rotation_to_vector(orig)
            ox, oy, _ = rotated
            twist_rad = math.radians(-twist)
            rx = ox * math.cos(twist_rad) - oy * math.sin(twist_rad)
            ry = ox * math.sin(twist_rad) + oy * math.cos(twist_rad)
            normal_ref = [rx, ry]
            final_orientation = [rx, ry, 0.0]
            logger.info(f"Web {web_name}: original {orig} → final_orientation {final_orientation} (after global transform + twist)")
        else:
            normal_ref = [1 if idx % 2 == 0 else -1, 0]
            final_orientation = [normal_ref[0], normal_ref[1], 0.0]
            logger.warning(f"No orientation for {web_name}, using fallback {final_orientation}")

        web_definition[web_name] = Web(
            coord_input=web_points_2d,
            plies=plies,
            normal_ref=normal_ref,
            orientation=final_orientation,
        )
        logger.info(f"Defined web {web_name}: {len(web_points_2d)} points, {len(plies)} plies")

    return skins, web_definition


def log_thicknesses(skins: dict, web_definition: dict) -> None:
    """Log thickness information."""
    logger.info(f"Assigned thickness arrays for airfoil skins: {list(skins.keys())}")
    for skin_name, skin in skins.items():
        if skin.thickness.array:
            logger.info(f"Skin {skin_name}: min {min(skin.thickness.array):.3f}, max {max(skin.thickness.array):.3f}")
    for web_name, web in web_definition.items():
        for i, ply in enumerate(web.plies):
            if ply.thickness.array:
                logger.info(f"Web {web_name} ply {i}: min {min(ply.thickness.array):.3f}, max {max(ply.thickness.array):.3f}")
