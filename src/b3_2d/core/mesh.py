"""Meshing functionality for b3_2d using cgfoil."""

from .mesh_processing import process_vtp_multi_section, process_single_section
from .mesh_extraction import (
    extract_airfoil_and_web_points,
    get_thickness_and_material_arrays,
    get_ply_thicknesses_and_materials,
    define_skins_and_webs,
    log_thicknesses,
)
from .mesh_utils import validate_points, sort_points_by_y, bb_size
from .bom import compute_bom
