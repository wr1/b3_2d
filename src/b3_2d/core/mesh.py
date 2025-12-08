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

logger = logging.getLogger(__name__)

# Rotation angle around z-axis
ROTATION_ANGLE = 90


def sort_points_by_y(mesh: pv.PolyData) -> pv.PolyData:
    """
    For a triangular PolyData mesh:
    - Sort points by y-coordinate (min to max)
    - Update triangle connectivity
    - Keep point_data correctly aligned
    """
    mesh = mesh.copy()
    n = mesh.n_points

    # Sort by y-coordinate
    sorted_indices = sorted(range(n), key=lambda i: mesh.points[i, 1])
    mesh.points = mesh.points[sorted_indices]

    # Create mapping from old to new indices
    old_to_new = {old: new for new, old in enumerate(sorted_indices)}

    # Update connectivity
    cells = mesh.cells.copy()
    cells[1::4] = [old_to_new[cells[i]] for i in range(1, len(cells), 4)]
    cells[2::4] = [old_to_new[cells[i]] for i in range(2, len(cells), 4)]
    cells[3::4] = [old_to_new[cells[i]] for i in range(3, len(cells), 4)]
    mesh.cells = cells

    # Update point_data
    for key in mesh.point_data.keys():
        mesh.point_data[key] = mesh.point_data[key][sorted_indices]

    return mesh


def validate_points(points_2d):
    """Validate that points_2d is a list of lists or tuples with 2 floats."""
    if not isinstance(points_2d, list):
        return False
    for p in points_2d:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            return False
        if not all(isinstance(c, (int, float)) for c in p):
            return False
    return True


def extract_airfoil_and_web_points(section_mesh):
    """Extract airfoil and web points from section mesh."""
    min_panel_id = section_mesh.cell_data["panel_id"].min()
    airfoil = pv.merge(
        [
            section_mesh.threshold(
                value=(0, section_mesh.cell_data["panel_id"].max()), scalars="panel_id"
            ),
            sort_points_by_y(
                section_mesh.threshold(
                    value=(min_panel_id, min_panel_id), scalars="panel_id"
                )
            ),
        ]
    )
    points_2d = airfoil.points[:, :2].tolist()

    web1 = section_mesh.threshold(value=(-1, -1), scalars="panel_id")
    web2 = section_mesh.threshold(value=(-2, -2), scalars="panel_id")

    web_points_2d_1 = web1.points[:, :2].tolist()
    web_points_2d_2 = web2.points[:, :2].tolist()

    web_data = [(web_points_2d_1, web1), (web_points_2d_2, web2)]

    return points_2d, web_data, airfoil


def get_thickness_arrays(mesh):
    """Get thickness arrays from mesh."""
    mesh_point = mesh.cell_data_to_point_data()
    thickness_keys = [
        k for k in mesh_point.point_data.keys() if re.match(r"ply_.*_thickness", k)
    ]
    thickness_keys.sort(key=lambda x: int(re.search(r"ply_(\d+)", x).group(1)))
    return {k: mesh_point.point_data[k] for k in thickness_keys}


def get_ply_thicknesses(airfoil, web_data):
    """Get ply thicknesses from mesh data."""
    airfoil_thicknesses = get_thickness_arrays(airfoil)

    web_thicknesses = []
    for _, web_mesh in web_data:
        web_thicknesses.append(get_thickness_arrays(web_mesh))

    return airfoil_thicknesses, web_thicknesses


def define_skins_and_webs(airfoil_thicknesses, web_data, web_thicknesses):
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


def log_thicknesses(skins, web_definition):
    """Log thickness information for skins and webs."""
    logger.info(f"Assigned thickness arrays for airfoil skins: {list(skins.keys())}")
    for skin_name, skin in skins.items():
        thickness = skin.thickness.array
        if thickness:
            logger.info(
                f"Skin {skin_name}: min thickness {min(thickness)}, max thickness {max(thickness)}"
            )
    logger.info(f"Assigned thickness arrays for webs: {list(web_definition.keys())}")
    for web_name, web in web_definition.items():
        for i, ply in enumerate(web.plies):
            thickness = ply.thickness.array
            if thickness:
                logger.info(
                    f"Web {web_name} ply {i} (material {ply.material}): min thickness {min(thickness)}, max thickness {max(thickness)}"
                )


def process_single_section(section_id, vtp_file, output_base_dir):
    """Process a single section_id."""
    section_dir = os.path.join(output_base_dir, f"section_{section_id}")
    os.makedirs(section_dir, exist_ok=True)

    # Temporarily set root logger level to WARNING to suppress stdout logging
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.WARNING)

    # Set up logging to section-specific file
    section_log_file = os.path.join(section_dir, "2dmesh.log")
    section_file_handler = logging.FileHandler(section_log_file)
    section_file_handler.setLevel(logging.INFO)
    logger.addHandler(section_file_handler)

    try:
        logger.info(f"Starting processing section_id: {section_id}")
        # Load VTP file in each process to avoid serialization issues
        mesh_vtp = pv.read(vtp_file).rotate_z(-90).rotate_x(180)

        # Filter mesh for this section_id
        section_mesh = mesh_vtp.threshold(
            value=(section_id, section_id), scalars="section_id"
        )

        # Extract points
        points_2d, web_data, airfoil = extract_airfoil_and_web_points(section_mesh)

        if not points_2d or len(points_2d) < 10 or not validate_points(points_2d):
            logger.warning(
                f"Invalid or insufficient airfoil points for section {section_id}, skipping"
            )
            return

        # Get ply thicknesses
        airfoil_thicknesses, web_thicknesses = get_ply_thicknesses(airfoil, web_data)
        if not airfoil_thicknesses:
            logger.warning(
                f"No ply thicknesses found for section {section_id}, skipping"
            )
            return

        # Define skins and webs
        skins, web_definition = define_skins_and_webs(
            airfoil_thicknesses, web_data, web_thicknesses
        )

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
    finally:
        # Remove the section-specific file handler
        logger.removeHandler(section_file_handler)
        # Restore root logger level
        root_logger.setLevel(original_level)


def process_vtp_multi_section(
    vtp_file: str, output_base_dir: str, num_processes: int = None
):
    """Process VTP file for all unique section_ids, outputting to subdirectories, using multiprocessing."""
    mesh_vtp = pv.read(vtp_file).rotate_z(ROTATION_ANGLE)

    # Get unique section_ids
    if "section_id" not in mesh_vtp.cell_data:
        raise ValueError("section_id not found in VTP file")
    unique_section_ids = mesh_vtp.cell_data["section_id"]
    unique_ids = sorted(set(unique_section_ids))
    total_sections = len(unique_ids)
    logger.info(f"Found {total_sections} unique section_ids: {np.array(unique_ids)}")

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    # Process sections in parallel with a wait graphic
    with Progress() as progress:
        spinner = progress.add_task("Processing sections...", total=None)
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(
                process_single_section,
                [(section_id, vtp_file, output_base_dir) for section_id in unique_ids],
            )
        progress.update(spinner, completed=True)
