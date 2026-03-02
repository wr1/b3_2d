"""Meshing functionality for b3_2d using cgfoil."""

try:
    from .mesh_processing import process_vtp_multi_section, process_single_section
    from .mesh_extraction import (
        extract_airfoil_and_web_points,
        get_thickness_and_material_arrays,
        get_ply_thicknesses_and_materials,
        define_skins_and_webs,
        log_thicknesses,
    )
    from .bom import compute_bom
except ImportError:
    process_vtp_multi_section = None
    process_single_section = None
    extract_airfoil_and_web_points = None
    get_thickness_and_material_arrays = None
    get_ply_thicknesses_and_materials = None
    define_skins_and_webs = None
    log_thicknesses = None
    compute_bom = None
