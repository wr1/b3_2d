"""Mesh extraction utilities for b3_2d."""

import logging
import re
import numpy as np
import pyvista as pv
from cgfoil.models import Skin, Web, Ply, Thickness

from .mesh_utils import bb_size, sort_points_by_y

logger = logging.getLogger(__name__)


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

    min_panel_id = min(negative_panel_ids)  # TE
    web_panel_ids = sorted(
        [pid for pid in negative_panel_ids if pid != min_panel_id], reverse=True
    )  # -1, -2, ...

    te = sort_points_by_y(
        section_mesh.threshold(value=(min_panel_id, min_panel_id), scalars="panel_id")
    )
    section = section_mesh.threshold(value=(0, panel_ids.max()), scalars="panel_id")
    section_bb = bb_size(section)
    te_bb = bb_size(te)
    if te_bb > 0.03 * section_bb:
        airfoil = pv.merge([section, te])
    else:
        airfoil = section
    points_2d = airfoil.points[:, :2].tolist()
    logger.info(f"Extracted {len(points_2d)} airfoil points")

    web_data = []
    for pid in web_panel_ids:
        web = section_mesh.threshold(value=(pid, pid), scalars="panel_id")
        if web.n_cells == 0:
            logger.warning(f"No cells for web panel_id={pid}")
            continue
        web_points_2d = web.points[:, :2].tolist()
        web_data.append((web_points_2d, web))
        logger.info(f"Web for panel_id={pid}: {len(web_points_2d)} points")
    logger.info(f"Extracted {len(web_data)} webs with panel_ids: {web_panel_ids}")
    return points_2d, web_data, airfoil


def get_thickness_and_material_arrays(mesh: pv.PolyData) -> tuple:
    """Get thickness and material arrays from mesh.

    Snaps averaged point data for thicknesses back to original discrete cell levels
    to eliminate averaging smearing (e.g. artificial 0.004 values between 0.0 and 0.008).
    Materials stay cell-based (max value per panel).
    """
    # Thickness keys from ORIGINAL cell_data (discrete levels)
    thickness_keys = [k for k in mesh.cell_data if re.match(r"ply_.*_thickness", k)]
    thickness_keys.sort(key=lambda x: int(re.search(r"ply_(\d+)", x).group(1)))
    material_keys = [k.replace("_thickness", "_material") for k in thickness_keys]

    logger.info(f"Thickness keys: {thickness_keys}")
    logger.info(f"Material keys: {material_keys}")
    logger.info(f"Available cell data keys: {list(mesh.cell_data.keys())}")

    mesh_point = mesh.cell_data_to_point_data(pass_cell_data=True)

    thicknesses = {}
    for k in thickness_keys:
        cell_thick = mesh.cell_data[k]
        # Unique original levels (rounded to remove float noise)
        unique_levels = np.sort(np.unique(np.round(cell_thick, decimals=8)))
        point_thick = mesh_point.point_data[k]

        # Snap every nodal value to nearest original discrete level
        snapped = np.array(
            [unique_levels[np.argmin(np.abs(unique_levels - v))] for v in point_thick]
        )

        unique, counts = np.unique(snapped, return_counts=True)
        logger.info(
            f"{k} after snap (no smear): len={len(snapped)}, "
            f"unique={dict(zip(unique.tolist(), counts.tolist()))}"
        )

        thicknesses[k] = snapped.tolist()

    materials = {k: mesh.cell_data[k] for k in material_keys}
    return thicknesses, materials


def get_ply_thicknesses_and_materials(airfoil: pv.PolyData, web_data: list) -> tuple:
    """Get ply thicknesses and materials from airfoil and web data."""
    airfoil_thicknesses, airfoil_materials = get_thickness_and_material_arrays(airfoil)
    web_thicknesses_and_materials = [
        get_thickness_and_material_arrays(web_mesh) for _, web_mesh in web_data
    ]
    web_thicknesses = [t for t, _ in web_thicknesses_and_materials]
    web_materials = [m for _, m in web_thicknesses_and_materials]
    return airfoil_thicknesses, airfoil_materials, web_thicknesses, web_materials


def define_skins_and_webs(
    airfoil_thicknesses: dict,
    airfoil_materials: dict,
    web_data: list,
    web_thicknesses: list,
    web_materials: list,
) -> tuple:
    """Define skins and webs from thicknesses, materials, and points."""
    # Skins
    skin_thickness_keys = sorted(
        airfoil_thicknesses,
        key=lambda k: int(re.search(r"ply_(\d+)", k).group(1)),
    )
    skins = {}
    for i, key in enumerate(skin_thickness_keys, 1):
        mat_key = key.replace("_thickness", "_material")
        mat_array = airfoil_materials[mat_key]
        material = int(np.max(mat_array))
        logger.info(
            f"For skin {i}, thickness key: {key}, material key: {mat_key}, "
            f"material: {material}"
        )
        skins[f"skin{i}"] = Skin(
            thickness=Thickness(type="array", array=list(airfoil_thicknesses[key])),
            material=material,
            sort_index=i,
        )

    # Webs
    web_definition = {}
    n_webs = len(web_data)
    web_names = [f"web{i + 1}" for i in range(n_webs)]
    for idx, web_name in enumerate(web_names):
        thicknesses = web_thicknesses[idx]
        materials = web_materials[idx]
        points = web_data[idx][0]
        ply_thickness_keys = sorted(
            thicknesses,
            key=lambda k: int(re.search(r"ply_(\d+)", k).group(1)),
        )
        plies = []
        for key in ply_thickness_keys:
            mat_key = key.replace("_thickness", "_material")
            mat_array = materials[mat_key]
            material = int(np.max(mat_array))
            plies.append(
                Ply(
                    thickness=Thickness(type="array", array=list(thicknesses[key])),
                    material=material,
                )
            )
        sign = 1 if idx % 2 == 0 else -1
        normal_ref = [sign, 0]
        web_definition[web_name] = Web(
            coord_input=points, plies=plies, normal_ref=normal_ref
        )
        logger.info(
            f"Defined web {web_name}: {len(points)} points, {len(plies)} plies, normal_ref={normal_ref}"
        )
    return skins, web_definition


def log_thicknesses(skins: dict, web_definition: dict) -> None:
    """Log thickness information."""
    logger.info(f"Assigned thickness arrays for airfoil skins: {list(skins.keys())}")
    for skin_name, skin in skins.items():
        thickness = skin.thickness.array
        if thickness:
            logger.info(
                f"Skin {skin_name}: min {min(thickness):.3f}, max {max(thickness):.3f}"
            )
    logger.info(f"Assigned thickness arrays for webs: {list(web_definition.keys())}")
    for web_name, web in web_definition.items():
        for i, ply in enumerate(web.plies):
            thickness = ply.thickness.array
            if thickness:
                logger.info(
                    f"Web {web_name} ply {i}: min {min(thickness):.3f}, "
                    f"max {max(thickness):.3f}"
                )
