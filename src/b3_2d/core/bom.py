"""Bill of Materials (BOM) calculations for b3_2d."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def compute_bom(mesh, matdb: dict = None):
    """Compute BOM from mesh Area and material_id, including mass if matdb provided."""
    if "Area" not in mesh.cell_data or "material_id" not in mesh.cell_data:
        return None
    
    total_area = float(mesh.cell_data["Area"].sum())
    areas_per_material = {}
    for mat_id in np.unique(mesh.cell_data["material_id"]):
        mask = mesh.cell_data["material_id"] == mat_id
        areas_per_material[int(mat_id)] = float(mesh.cell_data["Area"][mask].sum())
    
    bom_data = {"total_area": total_area, "areas_per_material": areas_per_material}
    
    if matdb:
        masses_per_material = {}
        total_mass = 0.0
        for mat_id, area in areas_per_material.items():
            if mat_id in matdb and "density" in matdb[mat_id]:
                density = matdb[mat_id]["density"]
                mass = area * density
                masses_per_material[mat_id] = mass
                total_mass += mass
            else:
                logger.warning(f"Density not found in matdb for material ID {mat_id}; skipping mass calculation.")
        bom_data["total_mass"] = total_mass
        bom_data["masses_per_material"] = masses_per_material
    
    return bom_data
