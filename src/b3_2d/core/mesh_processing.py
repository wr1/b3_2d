"""Mesh processing functionality for b3_2d."""

import os
import multiprocessing
import logging
import pickle
import numpy as np
import pyvista as pv
from rich.progress import Progress
from cgfoil.core.generate_mesh import generate_mesh
from cgfoil.models import AirfoilMesh
from cgfoil.cli.export import export_mesh_to_anba
from cgfoil.utils.io import save_mesh_to_vtk
from .mesh_extraction import (
    extract_airfoil_and_web_points,
    get_ply_thicknesses_and_materials,
    define_skins_and_webs,
    log_thicknesses,
)
from .mesh_utils import validate_points
from .plotting import plot_section_debug

logger = logging.getLogger(__name__)


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
