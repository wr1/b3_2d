"""Meshing functionality for b3_2d using cgfoil."""

import os
import multiprocessing
import re
import logging
import pyvista as pv
from cgfoil.core.main import run_cgfoil
from cgfoil.models import Skin, Web, Ply, AirfoilMesh, Thickness

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
        section_mesh = mesh_vtp.threshold(
            value=(section_id, section_id), scalars="section_id"
        )

        # Extract airfoil (assuming panel_id logic similar to original)
        airfoil = section_mesh.threshold(value=(0, 12), scalars="panel_id")
        te = section_mesh.threshold(value=(-3, -3), scalars="panel_id")
        points_2d = airfoil.points[:, :2].tolist()[:-1] + te.points[:, :2].tolist()[1:]
        points_2d = [tuple(p) for p in points_2d]  # Convert to list of tuples for validation

        if not points_2d or len(points_2d) < 10 or not validate_points(points_2d):
            logger.warning(f"Invalid or insufficient airfoil points for section {section_id}, skipping")
            return

        # Extract webs
        web1 = section_mesh.threshold(value=(-1, -1), scalars="panel_id")
        web2 = section_mesh.threshold(value=(-2, -2), scalars="panel_id")
        web_points_2d_1 = web1.points[:, :2].tolist()
        web_points_2d_2 = web2.points[:, :2].tolist()
        web_points_2d_1 = [tuple(p) for p in web_points_2d_1]  # Convert to list of tuples
        web_points_2d_2 = [tuple(p) for p in web_points_2d_2]  # Convert to list of tuples

        # Get all ply thickness fields dynamically
        airfoil_point_data = airfoil.cell_data_to_point_data().point_data
        te_point_data = te.cell_data_to_point_data().point_data
        ply_fields = [
            f
            for f in airfoil_point_data.keys()
            if re.match(r"ply_\d+_\w+_\d+_thickness", f)
        ]

        ply_thicknesses = {}
        for field in ply_fields:
            if field in te_point_data:
                airfoil_thickness = list(airfoil_point_data[field] * 0.01 + 0.04)[:-1]
                te_thickness = list(te_point_data[field] * 1 + 0.04)[1:]
                combined_thickness = airfoil_thickness + te_thickness
                match = re.match(r"ply_(\d+)_(\w+)_(\d+)_thickness", field)
                if match:
                    num1, slab, num2 = match.groups()
                    key = f"{num1}_{slab}_{num2}"
                    ply_thicknesses[key] = combined_thickness

        if not ply_thicknesses:
            logger.warning(f"No ply thicknesses found for section {section_id}, skipping")
            return

        # Define skins dynamically
        skins = {}
        for i, (key, thickness) in enumerate(ply_thicknesses.items()):
            skins[f"skin_{i}"] = Skin(
                thickness=Thickness(type="array", array=thickness),
                material=i + 1,  # Sequential materials starting from 1
                sort_index=i,
            )

        # Define webs using full mesh like in cgfoil multi vtp example
        # Use coord_input with the full list of points, and assigned arrays for thickness
        web_definition = {}
        if web_points_2d_1 and len(web_points_2d_1) >= 2 and validate_points(web_points_2d_1):
            web_thickness = [0.004] * len(web_points_2d_1)  # Assigned array, constant for simplicity
            web_definition["web1"] = Web(
                coord_input=web_points_2d_1,  # Full list of points
                plies=[
                    Ply(
                        thickness=Thickness(type="array", array=web_thickness),
                        material=len(skins) + 1,
                    ),
                ],
                normal_ref=[1, 0],
                n_elem=None,
            )
        if web_points_2d_2 and len(web_points_2d_2) >= 2 and validate_points(web_points_2d_2):
            web_thickness = [0.004] * len(web_points_2d_2)  # Assigned array, constant for simplicity
            web_definition["web2"] = Web(
                coord_input=web_points_2d_2,  # Full list of points
                plies=[
                    Ply(
                        thickness=Thickness(type="array", array=web_thickness),
                        material=len(skins) + 2,
                    ),
                ],
                normal_ref=[-1, 0],
                n_elem=None,
            )

        # Create AirfoilMesh
        mesh = AirfoilMesh(
            skins=skins,
            webs=web_definition,
            airfoil_input=points_2d,  # Use the extracted points as list of tuples
            n_elem=None,  # Do not set to avoid remeshing and messing with thickness distribution
            plot=False,  # Disable plotting in parallel to avoid issues
            plot_filename=None,
            vtk=os.path.join(section_dir, "output.vtk"),
        )

        # Run the meshing
        run_cgfoil(mesh)
        logger.info(f"Completed processing section_id: {section_id}")
    except Exception as e:
        logger.error(f"Error processing section_id {section_id}: {e}")


def process_vtp_multi_section(
    vtp_file: str, output_base_dir: str, num_processes: int = None
):
    """Process VTP file for all unique section_ids, outputting to subdirectories, using multiprocessing."""
    # Load VTP file to get unique ids
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
