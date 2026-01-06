"""Spanwise plotting utilities for b3_2d."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def plot_span_anba(output_dir: str, output_file: str) -> None:
    """Plot ANBA centers and angles along blade span."""
    anba_files = list(Path(output_dir).glob("section_*/anba_out.json"))
    data_list = []
    for af in anba_files:
        sid = int(af.parent.name.split("_")[1])
        with open(af, "r") as f:
            data = json.load(f)
        required_keys = ["mass_center", "shear_center", "tension_center", "principal_angle"]
        if all(k in data for k in required_keys):
            data_list.append((sid, data))
        else:
            logger.warning(f"Missing required keys in {af}")
    data_list.sort(key=lambda x: x[0])
    if not data_list:
        logger.warning("No valid ANBA data found")
        return
    z_vals = [d[0] for d in data_list]
    mass_centers = [d[1]["mass_center"] for d in data_list]
    shear_centers = [d[1]["shear_center"] for d in data_list]
    tension_centers = [d[1]["tension_center"] for d in data_list]
    principal_angles = [d[1]["principal_angle"] for d in data_list]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # X positions
    axes[0, 0].plot(z_vals, [mc[0] for mc in mass_centers], label="Mass Center X")
    axes[0, 0].plot(z_vals, [sc[0] for sc in shear_centers], label="Shear Center X")
    axes[0, 0].plot(z_vals, [tc[0] for tc in tension_centers], label="Tension Center X")
    axes[0, 0].set_xlabel("Section ID")
    axes[0, 0].set_ylabel("X Position")
    axes[0, 0].legend()
    axes[0, 0].set_title("X Positions along Blade")
    # Y positions
    axes[0, 1].plot(z_vals, [mc[1] for mc in mass_centers], label="Mass Center Y")
    axes[0, 1].plot(z_vals, [sc[1] for sc in shear_centers], label="Shear Center Y")
    axes[0, 1].plot(z_vals, [tc[1] for tc in tension_centers], label="Tension Center Y")
    axes[0, 1].set_xlabel("Section ID")
    axes[0, 1].set_ylabel("Y Position")
    axes[0, 1].legend()
    axes[0, 1].set_title("Y Positions along Blade")
    # Principal angle
    axes[1, 0].plot(z_vals, np.degrees(principal_angles))
    axes[1, 0].set_xlabel("Section ID")
    axes[1, 0].set_ylabel("Principal Angle (deg)")
    axes[1, 0].set_title("Principal Angle along Blade")
    # Leave bottom right empty
    plt.tight_layout()
    plt.savefig(output_file, dpi=400)
    plt.close()
    logger.info(f"Span plot saved to {output_file}")
