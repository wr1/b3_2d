"""Spanwise plotting utilities for b3_2d."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def plot_span_anba(output_dir: str, output_file: str) -> None:
    """Plot ANBA centers, angles, and stiffnesses/masses along blade span."""
    anba_files = list(Path(output_dir).glob("section_*/anba_out.json"))
    data_list = []
    has_stiff = False
    for af in anba_files:
        sid = int(af.parent.name.split("_")[1])
        with open(af, "r") as f:
            data = json.load(f)
        required_keys = ["mass_center", "shear_center", "tension_center", "principal_angle"]
        if all(k in data for k in required_keys):
            data_list.append((sid, data))
            if "Stiff" in data and "M" in data:
                has_stiff = True
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
    nrows = 4 if has_stiff else 2
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 6 * nrows // 2))
    if nrows == 2:
        axes = axes.reshape(2, 2)
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
    # Leave 1,1 empty for centers
    if has_stiff:
        stiffs = [d[1]["Stiff"] for d in data_list if "Stiff" in d[1]]
        masses = [d[1]["M"] for d in data_list if "M" in d[1]]
        z_stiff = [d[0] for d in data_list if "Stiff" in d[1]]
        # Stiff[0][0]
        axes[2, 0].plot(z_stiff, [s[0][0] for s in stiffs])
        axes[2, 0].set_xlabel("Section ID")
        axes[2, 0].set_ylabel("Stiff[0][0]")
        axes[2, 0].set_title("Stiffness [0][0] along Blade")
        # Stiff[1][1]
        axes[2, 1].plot(z_stiff, [s[1][1] for s in stiffs])
        axes[2, 1].set_xlabel("Section ID")
        axes[2, 1].set_ylabel("Stiff[1][1]")
        axes[2, 1].set_title("Stiffness [1][1] along Blade")
        # M[0][0]
        axes[3, 0].plot(z_stiff, [m[0][0] for m in masses])
        axes[3, 0].set_xlabel("Section ID")
        axes[3, 0].set_ylabel("M[0][0]")
        axes[3, 0].set_title("Mass [0][0] along Blade")
        # M[1][1]
        axes[3, 1].plot(z_stiff, [m[1][1] for m in masses])
        axes[3, 1].set_xlabel("Section ID")
        axes[3, 1].set_ylabel("M[1][1]")
        axes[3, 1].set_title("Mass [1][1] along Blade")
    plt.tight_layout()
    plt.savefig(output_file, dpi=400)
    plt.close()
    logger.info(f"Span plot saved to {output_file}")
