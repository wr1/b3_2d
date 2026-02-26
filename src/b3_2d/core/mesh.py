"""Meshing functionality for b3_2d using cgfoil."""

try:
    from .mesh_processing import process_vtp_multi_section
except ImportError:

    def process_vtp_multi_section(*args, **kwargs):
        raise ImportError("cgfoil not available")
