"""Meshing functionality for b3_2d using cgfoil."""

import os
import multiprocessing
import re
import logging
import pickle
import numpy as np
import pyvista as pv
from rich.progress import Progress
from cgfoil.core.main import generate_mesh
from cgfoil.models import Skin, Web, Ply, AirfoilMesh, Thickness
from cgfoil.cli.export import export_mesh_to_anba
from cgfoil.utils.io import save_mesh_to_vtk
from .plotting import plot_section_debug

logger = logging.getLogger(__name__)


def sort_points_by_y(mesh: pv.PolyData) -> pv.PolyData:
    """Sort points by y-coordinate and update connectivity."""
    mesh = mesh.copy()
    n = mesh.n_points
    sorted_indices = sorted(range(n), key=lambda i: mesh.points[i, 1])
    mesh.points = mesh.points[sorted_indices]
    old_to_new = {old: new for new, old in enumerate(sorted_indices)}
    cells = mesh.cells.copy()
    cells[1::4] = [old_to_new[cells[i]] for i in range(1, len(cells), 4)]
    cells[2::4] = [old_to_new[cells[i]] for i in range(2, len(cells), 4)]
    cells[3::4] = [old_to_new[cells[i]] for i in range(3, len(cells), 4)]
    mesh.cells = cells
    for key in mesh.point_data.keys():
        mesh.point_data[key] = mesh.point_data[key][sorted_indices]
    return mesh


def validate_points(points_2d: list) -> bool:
    """Validate points_2d is a list of 2D points."""
    if not isinstance(points_2d, list):
        return False
    for p in points_2d:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            return False
        if not all(isinstance(c, (int, float)) for c in p):
            return False
    return True


def bb_size(mesh: pv.PolyData) -> float:
    """Compute bounding box size."""
    bounds = mesh.bounds
    return ((bounds[1] - bounds[0]) ** 2 + (bounds[3] - bounds[2]) ** 2) ** 0.5


def extract_airfoil_and_web_points(section_mesh: pv.PolyData) -> tuple:
    """Extract airfoil and web points, applying de-offset and de-twist."""
    min_panel_id = section_mesh.cell_data["panel_id"].min()

    twist = np.unique(section_mesh.cell_data["twist"])[0]
    dx = np.unique(section_mesh.cell_data["dx"])[0]
    dy = np.unique(section_mesh.cell_data["dy"])[0]
    logger.info(f"Applying translation dx={dx}, dy={dy} and twist={twist} degrees")

    # section_mesh.points[:, 0] += dx
    # section_mesh.points[:, 1] += dy

    section_mesh.points[:, 0] -= dy
    section_mesh.points[:, 1] -= dx

    section_mesh = section_mesh.rotate_z(-twist)

    section = section_mesh.threshold(
        value=(0, section_mesh.cell_data["panel_id"].max()), scalars="panel_id"
    )
    te = sort_points_by_y(
        section_mesh.threshold(value=(min_panel_id, min_panel_id), scalars="panel_id")
    )
    section_bb = bb_size(section)
    te_bb = bb_size(te)
    if te_bb > 0.03 * section_bb:
        airfoil = pv.merge([section, te])
    else:
        airfoil = section
    points_2d = airfoil.points[:, :2].tolist()
    logger.info(f"Extracted {len(points_2d)} airfoil points")
    web1 = section_mesh.threshold(value=(-1, -1), scalars="panel_id")
    web2 = section_mesh.threshold(value=(-2, -2), scalars="panel_id")
    web_points_2d_1 = web1.points[:, :2].tolist()
    web_points_2d_2 = web2.points[:, :2].tolist()
    web_data = [(web_points_2d_1, web1), (web_points_2d_2, web2)]
    return points_2d, web_data, airfoil


def get_thickness_arrays(mesh: pv.PolyData) -> dict:
    """Get thickness arrays from mesh."""
    mesh_point = mesh.cell_data_to_point_data()
    thickness_keys = [
        k for k in mesh_point.point_data.keys() if re.match(r"ply_.*_thickness", k)
    ]
    thickness_keys.sort(key=lambda x: int(re.search(r"ply_(\d+)", x).group(1)))
    return {k: mesh_point.point_data[k] for k in thickness_keys}


def get_ply_thicknesses(airfoil: pv.PolyData, web_data: list) -> tuple:
    """Get ply thicknesses from airfoil and web data."""
    airfoil_thicknesses = get_thickness_arrays(airfoil)
    web_thicknesses = [get_thickness_arrays(web_mesh) for _, web_mesh in web_data]
    return airfoil_thicknesses, web_thicknesses


def define_skins_and_webs(
    airfoil_thicknesses: dict, web_data: list, web_thicknesses: list
) -> tuple:
    """Define skins and webs from thicknesses and points."""
    skins = {}
    material_id = 1
    for i, (key, thickness_array) in enumerate(airfoil_thicknesses.items(), start=1):
        skins[f"skin{i}"] = Skin(
            thickness=Thickness(type="array", array=list(thickness_array)),
            material=material_id,
            sort_index=i,
        )
        material_id += 1
    web_definition = {}
    web_meshes = [
        ("web1", web_thicknesses[0], web_data[0][0]),
        ("web2", web_thicknesses[1], web_data[1][0]),
    ]
    for web_name, thicknesses, points in web_meshes:
        plies = [
            Ply(
                thickness=Thickness(type="array", array=list(thicknesses[key])),
                material=material_id + j,
            )
            for j, key in enumerate(thicknesses)
        ]
        material_id += len(thicknesses)
        normal_ref = [1, 0] if web_name == "web1" else [-1, 0]
        web_definition[web_name] = Web(
            coord_input=points, plies=plies, normal_ref=normal_ref
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
                    f"Web {web_name} ply {i} (mat {ply.material}): min {min(thickness):.3f}, max {max(thickness):.3f}"
                )


def process_single_section(
    section_id: int, vtp_file: str, output_base_dir: str, debug: bool = False
) -> None:
    """Process a single section."""
    section_dir = os.path.join(output_base_dir, f"section_{section_id}")
    os.makedirs(section_dir, exist_ok=True)
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.WARNING)
    section_log_file = os.path.join(section_dir, "2dmesh.log")
    section_file_handler = logging.FileHandler(section_log_file)
    section_file_handler.setLevel(logging.INFO)
    logger.addHandler(section_file_handler)
    try:
        logger.info(f"Starting processing section {section_id}")
        mesh_vtp = pv.read(vtp_file).rotate_z(-90).rotate_x(180)
        section_mesh = mesh_vtp.threshold(
            value=(section_id, section_id), scalars="section_id"
        )
        if debug:
            plot_section_debug(section_mesh, section_dir, section_id)
        points_2d, web_data, airfoil = extract_airfoil_and_web_points(section_mesh)
        if not points_2d or len(points_2d) < 10 or not validate_points(points_2d):
            logger.warning(f"Invalid points for section {section_id}, skipping")
            return
        airfoil_thicknesses, web_thicknesses = get_ply_thicknesses(airfoil, web_data)
        if not airfoil_thicknesses:
            logger.warning(f"No thicknesses for section {section_id}, skipping")
            return
        skins, web_definition = define_skins_and_webs(
            airfoil_thicknesses, web_data, web_thicknesses
        )
        log_thicknesses(skins, web_definition)
        vtk_output_file = os.path.join(section_dir, "output.vtk")
        mesh = AirfoilMesh(
            skins=skins,
            webs=web_definition,
            airfoil_input=points_2d,
            n_elem=None,
            plot=False,
            plot_filename=None,
            vtk=vtk_output_file,
        )
        mesh_result = generate_mesh(mesh)
        if mesh.vtk:
            save_mesh_to_vtk(mesh_result, mesh, mesh.vtk)
        mesh_file = os.path.join(section_dir, "mesh.pck")
        with open(mesh_file, "wb") as f:
            pickle.dump(mesh_result, f)
        logger.info(f"Mesh saved to {mesh_file}")
        anba_file = os.path.join(section_dir, "anba.json")
        export_mesh_to_anba(mesh_file, anba_file)
        logger.info(f"Exported ANBA JSON to {anba_file}")
        logger.info(f"Completed section {section_id}")
    except Exception as e:
        logger.error(f"Error processing section {section_id}: {e}", exc_info=True)
    finally:
        logger.removeHandler(section_file_handler)
        root_logger.setLevel(original_level)


def process_vtp_multi_section(
    vtp_file: str, output_base_dir: str, num_processes: int = None, debug: bool = False
) -> None:
    """Process VTP file for all sections using multiprocessing."""
    mesh_vtp = pv.read(vtp_file)
    if "section_id" not in mesh_vtp.cell_data:
        raise ValueError("section_id not found in VTP file")
    unique_ids = sorted(set(mesh_vtp.cell_data["section_id"]))
    total_sections = len(unique_ids)
    logger.info(f"Found {total_sections} unique section_ids: {np.array(unique_ids)}")
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    with Progress() as progress:
        spinner = progress.add_task("Processing sections...", total=None)
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(
                process_single_section,
                [
                    (section_id, vtp_file, output_base_dir, debug)
                    for section_id in unique_ids
                ],
            )
        progress.update(spinner, completed=True)
