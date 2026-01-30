"""BOM plotting utilities for b3_2d."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def plot_bom_spanwise(output_dir: str, output_file: str, matdb: dict = None):
    """Plot BOM areas and masses along blade span."""
    bom_files = list(Path(output_dir).glob("section_*/bom.json"))
    data_list = []
    for bf in bom_files:
        sid = int(bf.parent.name.split("_")[1])
        with open(bf, "r") as f:
            data = json.load(f)
        required_keys = ["total_area", "areas_per_material"]
        if all(k in data for k in required_keys):
            data_list.append((sid, data))
        else:
            logger.warning(f"Missing required keys in {bf}")
    data_list.sort(key=lambda x: x[0])
    if not data_list:
        logger.warning("No valid BOM data found")
        return
    
    z_vals = [d[0] for d in data_list]
    total_areas = [d[1]["total_area"] for d in data_list]
    areas_per_material = [d[1]["areas_per_material"] for d in data_list]
    
    # Collect all material IDs
    all_mat_ids = set()
    for apm in areas_per_material:
        all_mat_ids.update(apm.keys())
    all_mat_ids = sorted(all_mat_ids)
    
    # Build ID to name mapping
    id_to_name = {}
    if matdb:
        for name, props in matdb.items():
            if 'id' in props:
                id_to_name[props['id']] = name
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Subplot 1: Areas
    axes[0].plot(z_vals, total_areas, label="Total Area", marker="o")
    for mat_id in all_mat_ids:
        mat_areas = [apm.get(mat_id, 0) for apm in areas_per_material]
        name = id_to_name.get(mat_id, f"ID {mat_id}")
        axes[0].plot(z_vals, mat_areas, label=f"{name} (ID {mat_id}) Area", marker="s")
    axes[0].set_xlabel("Section ID")
    axes[0].set_ylabel("Area")
    axes[0].legend()
    axes[0].set_title("Areas along Blade Span")
    axes[0].grid(True)
    
    # Subplot 2: Masses (if available)
    has_mass = any("total_mass" in d[1] for d in data_list)
    if has_mass:
        total_masses = [d[1].get("total_mass", 0) for d in data_list]
        masses_per_material = [d[1].get("masses_per_material", {}) for d in data_list]
        axes[1].plot(z_vals, total_masses, label="Total Mass", marker="o")
        for mat_id in all_mat_ids:
            mat_masses = [mpm.get(mat_id, 0) for mpm in masses_per_material]
            name = id_to_name.get(mat_id, f"ID {mat_id}")
            axes[1].plot(z_vals, mat_masses, label=f"{name} (ID {mat_id}) Mass", marker="s")
        axes[1].set_xlabel("Section ID")
        axes[1].set_ylabel("Mass")
        axes[1].legend()
        axes[1].set_title("Masses along Blade Span")
        axes[1].grid(True)
    else:
        axes[1].text(0.5, 0.5, "No mass data available", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("Masses along Blade Span (No Data)")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=400)
    plt.close()
    logger.info(f"BOM spanwise plot saved to {output_file}")
