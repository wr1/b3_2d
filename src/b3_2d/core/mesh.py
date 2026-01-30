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
    cells = mesh.cells.copy()
    old_to_new = {old: new for new, old in enumerate(sorted_indices)}
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
    """Get thickness and material arrays from mesh."""
    mesh_point = mesh.cell_data_to_point_data(pass_cell_data=True)
    thickness_keys = [
        k for k in mesh_point.point_data.keys() if re.match(r"ply_.*_thickness", k)
    ]
    thickness_keys.sort(key=lambda x: int(re.search(r"ply_(\d+)", x).group(1)))
    material_keys = [k.replace("_thickness", "_material") for k in thickness_keys]
    logger.info(f"Thickness keys: {thickness_keys}")
    logger.info(f"Material keys: {material_keys}")
    logger.info(f"Available point data keys: {list(mesh_point.point_data.keys())}")
    thicknesses = {k: mesh_point.point_data[k] for k in thickness_keys}
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


def process_single_section(
    section_id: int,
    vtp_file: str,
    output_base_dir: str,
    matdb: dict = None,
    debug: bool = False,
) -> dict:
    """Process a single section."""
    section_dir = os.path.join(output_base_dir, f"section_{section_id}")
    result = {
        "section_id": section_id,
        "success": True,
        "input_file": vtp_file,
        "output_dir": section_dir,
        "created_files": [],
        "errors": [],
    }
    os.makedirs(section_dir, exist_ok=True)
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.WARNING)
    section_log_file = os.path.join(section_dir, "2dmesh.log")
    result["created_files"].append(section_log_file)
    section_file_handler = logging.FileHandler(section_log_file)
    section_file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(lineno)d: %(message)s")
    section_file_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.propagate = False
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
            msg = f"Invalid points for section {section_id}, skipping"
            logger.warning(msg)
            result["success"] = False
            result["errors"].append(msg)
            return result
        airfoil_thicknesses, airfoil_materials, web_thicknesses, web_materials = (
            get_ply_thicknesses_and_materials(airfoil, web_data)
        )
        if not airfoil_thicknesses:
            msg = f"No thicknesses for section {section_id}, skipping"
            logger.warning(msg)
            result["success"] = False
            result["errors"].append(msg)
            return result
        skins, web_definition = define_skins_and_webs(
            airfoil_thicknesses,
            airfoil_materials,
            web_data,
            web_thicknesses,
            web_materials,
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
        logger.info(
            f"AirfoilMesh created with {len(skins)} skins and {len(web_definition)} webs"
        )
        mesh_result = generate_mesh(mesh)
        logger.info(f"Mesh generation completed, result type: {type(mesh_result)}")
        if mesh.vtk:
            save_mesh_to_vtk(mesh_result, mesh, mesh.vtk)
            logger.info(f"VTK file saved to {vtk_output_file}")
            # Debug: load and inspect VTK
            loaded_mesh = pv.read(vtk_output_file)
            logger.info(
                f"Loaded VTK: {loaded_mesh.n_cells} cells, {loaded_mesh.n_points} points"
            )
            if "material_id" in loaded_mesh.cell_data:
                unique_mats = np.unique(loaded_mesh.cell_data["material_id"])
                logger.info(f"Unique material_ids in VTK: {unique_mats}")
            else:
                logger.warning("No material_id in VTK cell_data")
            result["created_files"].append(vtk_output_file)
        mesh_file = os.path.join(section_dir, "mesh.pck")
        with open(mesh_file, "wb") as f:
            pickle.dump(mesh_result, f)
        anba_file = os.path.join(section_dir, "anba.json")
        export_mesh_to_anba(mesh_file, anba_file, matdb=matdb)
        os.remove(mesh_file)
        result["created_files"].append(anba_file)
        logger.info(f"Exported ANBA JSON to {anba_file}")
        logger.info(f"Completed section {section_id}")
        return result
    except Exception as e:
        msg = f"Error processing section {section_id}: {e}"
        logger.error(msg, exc_info=True)
        result["success"] = False
        result["errors"].append(str(e))
        return result
    finally:
        logger.removeHandler(section_file_handler)
        root_logger.setLevel(original_level)


def process_vtp_multi_section(
    vtp_file: str,
    output_base_dir: str,
    num_processes: int = None,
    matdb: dict = None,
    debug: bool = False,
) -> list[dict]:
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
            results = pool.starmap(
                process_single_section,
                [
                    (section_id, vtp_file, output_base_dir, matdb, debug)
                    for section_id in unique_ids
                ],
            )
        progress.update(spinner, completed=True)
    successful_count = sum(1 for r in results if r["success"])
    failed_count = len(results) - successful_count
    logger.info(
        f"Processed {total_sections} sections: {successful_count} successful, "
        f"{failed_count} failed."
    )
    if failed_count > 0:
        logger.warning("Failed sections:")
        for r in results:
            if not r["success"]:
                errors_str = "; ".join(r.get("errors", ["Unknown error"]))
                logger.warning(f"  Section {r['section_id']}: {errors_str}")
                if r["created_files"]:
                    logger.info(f"    Created partial files: {r['created_files']}")
    return results
