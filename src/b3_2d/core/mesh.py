"""Meshing functionality for b3_2d using cgfoil."""

import os
import multiprocessing
import re
from cgfoil.core.main import run_cgfoil
from cgfoil.models import Skin, Web, Ply, AirfoilMesh, Thickness

try:
    import pyvista as pv
except ImportError:
    raise ImportError("pyvista is required for b3_2d")

# Rotation angle around z-axis
ROTATION_ANGLE = 90


def process_single_section(args):
    """Process a single section_id."""
    section_id, vtp_file, output_base_dir = args
    try:
        print(f"Starting processing section_id: {section_id}")
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

        # Extract webs
        web1 = section_mesh.threshold(value=(-1, -1), scalars="panel_id")
        web2 = section_mesh.threshold(value=(-2, -2), scalars="panel_id")
        web_points_2d_1 = web1.points[:, :2].tolist()
        web_points_2d_2 = web2.points[:, :2].tolist()

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

        # Define skins dynamically
        skins = {}
        for i, (key, thickness) in enumerate(ply_thicknesses.items()):
            skins[f"skin_{i}"] = Skin(
                thickness=Thickness(type="array", array=thickness),
                material=i + 1,  # Sequential materials starting from 1
                sort_index=i,
            )

        # Define webs
        web_definition = {
            "web1": Web(
                points=web_points_2d_1,
                plies=[
                    Ply(
                        thickness=Thickness(type="constant", value=0.004),
                        material=len(skins) + 1,
                    ),
                ],
                normal_ref=[1, 0],
                n_cell=10,
            ),
            "web2": Web(
                points=web_points_2d_2,
                plies=[
                    Ply(
                        thickness=Thickness(type="constant", value=0.004),
                        material=len(skins) + 2,
                    ),
                ],
                normal_ref=[-1, 0],
                n_cell=10,
            ),
        }

        # Create AirfoilMesh
        mesh = AirfoilMesh(
            skins=skins,
            webs=web_definition,
            airfoil_input=points_2d,
            n_elem=None,
            plot=False,  # Disable plotting in parallel to avoid issues
            plot_filename=None,
            vtk=os.path.join(section_dir, "output.vtk"),
        )

        # Run the meshing
        run_cgfoil(mesh)
        print(f"Completed processing section_id: {section_id}")
    except Exception as e:
        print(f"Error processing section_id {section_id}: {e}")


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
    print(f"Found {total_sections} unique section_ids: {unique_ids}")

    # Prepare arguments for multiprocessing
    args_list = [(section_id, vtp_file, output_base_dir) for section_id in unique_ids]

    # Use multiprocessing Pool
    if num_processes is None:
        num_processes = min(multiprocessing.cpu_count(), total_sections)
    print(f"Using {num_processes} processes")
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_single_section, args_list)
