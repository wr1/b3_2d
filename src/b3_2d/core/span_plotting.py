"""Spanwise plotting utilities for b3_2d."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def plot_span_anba(output_dir: str, output_file: str) -> None:
    """Plot ANBA stiffnesses and masses along blade span."""
    anba_files = list(Path(output_dir).glob("section_*/anba_out.json"))
    data_list = []
    for af in anba_files:
        sid = int(af.parent.name.split("_")[1])
        with open(af, "r") as f:
            data = json.load(f)
        if "output_data" in data and "Stiff" in data["output_data"] and "M" in data["output_data"]:
            data_list.append((sid, data))
        else:
            logger.warning(f"Missing output_data or Stiff/M in {af}")
    data_list.sort(key=lambda x: x[0])
    if not data_list:
        logger.warning("No valid ANBA data found")
        return
    z_vals = [d[0] for d in data_list]
    stiffs = [d[1]["output_data"]["Stiff"] for d in data_list]
    masses = [d[1]["output_data"]["M"] for d in data_list]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # Stiff[0][0]
    axes[0, 0].plot(z_vals, [s[0][0] for s in stiffs])
    axes[0, 0].set_xlabel("Section ID")
    axes[0, 0].set_ylabel("Stiff[0][0]")
    axes[0, 0].set_title("Stiffness [0][0] along Blade")
    # Stiff[1][1]
    axes[0, 1].plot(z_vals, [s[1][1] for s in stiffs])
    axes[0, 1].set_xlabel("Section ID")
    axes[0, 1].set_ylabel("Stiff[1][1]")
    axes[0, 1].set_title("Stiffness [1][1] along Blade")
    # M[0][0]
    axes[1, 0].plot(z_vals, [m[0][0] for m in masses])
    axes[1, 0].set_xlabel("Section ID")
    axes[1, 0].set_ylabel("M[0][0]")
    axes[1, 0].set_title("Mass [0][0] along Blade")
    # M[1][1]
    axes[1, 1].plot(z_vals, [m[1][1] for m in masses])
    axes[1, 1].set_xlabel("Section ID")
    axes[1, 1].set_ylabel("M[1][1]")
    axes[1, 1].set_title("Mass [1][1] along Blade")
    plt.tight_layout()
    plt.savefig(output_file, dpi=400)
    plt.close()
    logger.info(f"Span plot saved to {output_file}")
