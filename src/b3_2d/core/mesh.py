"""Meshing functionality for b3_2d using cgfoil."""

import os
import multiprocessing
import re
import logging
import pickle
import math
import numpy as np
import pyvista as pv
from cgfoil.core.main import generate_mesh
from cgfoil.models import Skin, Web, Ply, AirfoilMesh, Thickness
from cgfoil.cli.export import export_mesh_to_anba
from cgfoil.utils.io import save_mesh_to_vtk

logger = logging.getLogger(__name__)

# Rotation angle around z-axis
ROTATION_ANGLE = 90


def validate_points(points_2d):
    """Validate that points_2d is a list of tuples with 2 floats."""
    if not isinstance(points_2d, list):
        return False
    for p in points_2d:
        if not isinstance(p, tuple) or len(p) != 2:
            return False
        if not all(isinstance(c, (int, float)) for c in p):
            return False
    return True


def extract_airfoil_and_web_points(section_mesh):
    """Extract airfoil and web points from section mesh."""
    airfoil = section_mesh.threshold(value=(0, 12), scalars="panel_id")
    te = section_mesh.threshold(value=(-3, -3), scalars="panel_id")
    points_2d = airfoil.points[:, :2].tolist()[:-1] + te.points[:, :2].tolist()[1:]
    points_2d = [tuple(p) for p in points_2d]
    web_ids = [-1, -2]
    web_data = []
    for web_id in web_ids:
        web_mesh = section_mesh.threshold(value=(web_id, web_id), scalars="panel_id")
        web_points = [tuple(p) for p in web_mesh.points[:, :2].tolist()]
        web_data.append((web_points, web_mesh))
    return points_2d, web_data, airfoil, te


def get_ply_thicknesses(airfoil, te, web_data):
    """Get ply thicknesses from mesh data."""
    airfoil_point_data = airfoil.cell_data_to_point_data().point_data
    te_point_data = te.cell_data_to_point_data().point_data
    ply_fields = [
        f for f in airfoil_point_data.keys() if re.match(r"ply_\d+_\w+_\d+_thickness", f)
    ]
    ply_thicknesses = {}
    for field in ply_fields:
        if field in te_point_data:
            airfoil_thickness = list(airfoil_point_data[field])[:-1]
            te_thickness = list(te_point_data[field])[1:]
            combined_thickness = airfoil_thickness + te_thickness
            match = re.match(r"ply_(\d+)_(\w+)_(\d+)_thickness", field)
            if match:
                num1, slab, num2 = match.groups()
                key = f"{num1}_{slab}_{num2}"
                ply_thicknesses[key] = combined_thickness
    web_plies_data = []
    for web_points, web_mesh in web_data:
        point_data = web_mesh.cell_data_to_point_data().point_data
        plies = []
        for field in ply_fields:
            if field in point_data:
                thickness_array = list(point_data[field])
                if any(t > 0 for t in thickness_array):
                    plies.append(
                        Ply(thickness=Thickness(type="array", array=thickness_array), material=0)  # material set later
                    )
        web_plies_data.append(plies)
    return ply_thicknesses, ply_fields, web_plies_data


def define_skins_and_webs(ply_thicknesses, web_data, web_plies_data):
    """Define skins and webs from thicknesses and points."""
    skins = {}
    for i, (key, thickness) in enumerate(ply_thicknesses.items()):
        skins[f"skin_{i}"] = Skin(
            thickness=Thickness(type="array", array=thickness),
            material=i + 1,
            sort_index=i,
        )
    web_definition = {}
    material_counter = len(skins) + 1
    for i, ((web_points, _), plies) in enumerate(zip(web_data, web_plies_data)):
        if web_points and len(web_points) >= 2 and validate_points(web_points) and plies:
            for ply in plies:
                ply.material = material_counter
                material_counter += 1
            normal_ref = [1, 0] if i == 0 else [-1, 0]  # Adjust for more webs if needed
            web_definition[f"web{i+1}"] = Web(points=web_points, plies=plies, normal_ref=normal_ref, n_elem=None)
    return skins, web_definition


def log_thicknesses(skins, web_definition):
    """Log thickness information for skins and webs."""
    logger.info(f"Assigned thickness arrays for airfoil skins: {list(skins.keys())}")
    for skin_name, skin in skins.items():
        thickness = skin.thickness.array
        if thickness:
            logger.info(f"Skin {skin_name}: min thickness {min(thickness)}, max thickness {max(thickness)}")
    logger.info(f"Assigned thickness arrays for webs: {list(web_definition.keys())}")
    for web_name, web in web_definition.items():
        for i, ply in enumerate(web.plies):
            thickness = ply.thickness.array
            if thickness:
                logger.info(f"Web {web_name} ply {i} (material {ply.material}): min thickness {min(thickness)}, max thickness {max(thickness)}")


def process_single_section(args):
    """Process a single section_id."""
    section_id, vtp_file, output_base_dir = args
    try:
        logger.info(f"Starting processing section_id: {section_id}")
        # Load VTP file in each process to avoid serialization issues
        mesh_vtp = pv.read(vtp_file).rotate_z(ROTATION_ANGLE)

        # Create subdirectory
        section_dir = os.path.join(output_base_dir, f"section_{section_id}")
        os.makedirs(section_dir, exist_ok=True)

        # Filter mesh for this section_id
        section_mesh = mesh_vtp.threshold(value=(section_id, section_id), scalars="section_id")

        # Extract points
        points_2d, web_data, airfoil, te = extract_airfoil_and_web_points(section_mesh)

        if not points_2d or len(points_2d) < 10 or not validate_points(points_2d):
            logger.warning(f"Invalid or insufficient airfoil points for section {section_id}, skipping")
            return

        # Get ply thicknesses
        ply_thicknesses, ply_fields, web_plies_data = get_ply_thicknesses(airfoil, te, web_data)
        if not ply_thicknesses:
            logger.warning(f"No ply thicknesses found for section {section_id}, skipping")
            return

        # Define skins and webs
        skins, web_definition = define_skins_and_webs(ply_thicknesses, web_data, web_plies_data)

        # Log thicknesses
        log_thicknesses(skins, web_definition)

        # Create AirfoilMesh
        mesh = AirfoilMesh(
            skins=skins,
            webs=web_definition,
            airfoil_input=points_2d,
            n_elem=None,
            plot=False,
            plot_filename=None,
            vtk=os.path.join(section_dir, "output.vtk"),
        )

        # Run the meshing
        mesh_result = generate_mesh(mesh)
        logger.info(f"Cross-sectional areas: {mesh_result.areas}")

        # Save VTK
        if mesh.vtk:
            save_mesh_to_vtk(mesh_result, mesh, mesh.vtk)

        # Save mesh
        mesh_file = os.path.join(section_dir, "mesh.pck")
        with open(mesh_file, "wb") as f:
            pickle.dump(mesh_result, f)
        logger.info(f"Mesh saved to {mesh_file}")

        # Export to ANBA
        anba_file = os.path.join(section_dir, "anba.json")
        export_mesh_to_anba(mesh_file, anba_file)
        logger.info(f"Exported ANBA JSON for section_id {section_id} to {section_dir}")

        logger.info(f"Completed processing section_id: {section_id}")
    except Exception as e:
        logger.error(f"Error processing section_id {section_id}: {e}", exc_info=True)


def process_vtp_multi_section(vtp_file: str, output_base_dir: str, num_processes: int = None):
    """Process VTP file for all unique section_ids, outputting to subdirectories, using multiprocessing."""
    mesh_vtp = pv.read(vtp_file).rotate_z(ROTATION_ANGLE)

    # Get unique section_ids
    if "section_id" not in mesh_vtp.cell_data:
        raise ValueError("section_id not found in VTP file")
    unique_section_ids = mesh_vtp.cell_data["section_id"]
    unique_ids = sorted(set(unique_section_ids))
    total_sections = len(unique_ids)
    logger.info(f"Found {total_sections} unique section_ids: {unique_ids}")

    # Prepare arguments for multiprocessing
    args_list = [(section_id, vtp_file, output_base_dir) for section_id in unique_ids]

    # Use multiprocessing Pool
    if num_processes is None:
        num_processes = min(multiprocessing.cpu_count(), total_sections)
    logger.info(f"Using {num_processes} processes")
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_single_section, args_list)
