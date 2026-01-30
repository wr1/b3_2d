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
        # Build lookup from id to properties
        id_to_props = {}
        for name, props in matdb.items():
            if "id" in props:
                id_to_props[props["id"]] = props

        masses_per_material = {}
        total_mass = 0.0
        for mat_id, area in areas_per_material.items():
            if mat_id in id_to_props and "rho" in id_to_props[mat_id]:
                density = id_to_props[mat_id]["rho"]
                mass = area * density
                masses_per_material[mat_id] = mass
                total_mass += mass
            else:
                if mat_id in id_to_props:
                    available_keys = list(id_to_props[mat_id].keys())
                    logger.warning(
                        f"Density (rho) not found in matdb for material ID {mat_id} (attempted key: 'rho'). Available keys in matdb for this material: {available_keys}. Skipping mass calculation."
                    )
                else:
                    available_mat_ids = list(id_to_props.keys())
                    logger.warning(
                        f"Material ID {mat_id} not found in matdb. Available material IDs: {available_mat_ids}. Skipping mass calculation."
                    )
        bom_data["total_mass"] = total_mass
        bom_data["masses_per_material"] = masses_per_material

    return bom_data
