import json
from pathlib import Path
from rich.table import Table
from rich.console import Console
import pyvista as pv
from statesman import Statesman, ManagedFile
from ..core.mesh import process_vtp_multi_section
from ..core.bom import compute_bom


class B32dStep(Statesman):
    """Statesman step for 2D meshing using cgfoil."""

    workdir_key = "workdir"
    input_files = [
        ManagedFile(name="b3_drp/draped.vtk", non_empty=True),
    ]
    output_files = ["b3_2d/"]  # Directory with outputs
    dependent_sections = ["draping"]  # Assuming draping section

    def _execute(self):
        """Execute the step."""
        self.logger.info("Executing B32dStep: 2D meshing with cgfoil.")
        config_dir = Path(self.config_path).parent
        workdir = config_dir / self.config["workdir"]
        vtp_file = workdir / "b3_drp" / "draped.vtk"
        output_dir = workdir / "b3_2d"
        output_dir.mkdir(parents=True, exist_ok=True)
        num_processes = self.config.get("num_processes", None)
        matdb = self.config.get("matdb", {})
        # Extract web orientations from config
        webs = self.config.get("structure", {}).get("webs", [])
        orientations = {}
        for w in webs:
            if w.get("mesh", False):
                name = w["name"]
                if "reference_web" in w:
                    # For ribbons, inherit from reference
                    ref = w["reference_web"]
                    if ref in orientations:
                        orientations[name] = orientations[ref]
                    else:
                        self.logger.warning(f"Reference web {ref} not found for {name}")
                        orientations[name] = [0, 1, 0]  # Default
                else:
                    orientations[name] = w.get("orientation", [0, 1, 0])
        results = process_vtp_multi_section(
            str(vtp_file), str(output_dir), num_processes, matdb=matdb, orientations=orientations
        )
        # Sort results by section_id for ordered display
        results = sorted(results, key=lambda x: x["section_id"])
        # Compute BOM for successful sections
        bom_sections = []
        for r in results:
            if r["success"]:
                vtk_file = Path(r["output_dir"]) / "output.vtk"
                if vtk_file.exists():
                    mesh = pv.read(str(vtk_file))
                    bom_data = compute_bom(mesh, matdb)
                    if bom_data:
                        bom_file = Path(r["output_dir"]) / "bom.json"
                        with open(bom_file, "w") as f:
                            json.dump(bom_data, f, indent=2)
                        r["created_files"].append(str(bom_file))
                        bom_sections.append(int(r["section_id"]))
        if bom_sections:
            self.logger.info(f"BOM computed for sections: {bom_sections}")
        console = Console()
        table = Table(title="Section Processing Results")
        table.add_column("Section ID", justify="right")
        table.add_column("Success")
        table.add_column("Errors")
        table.add_column("Path")
        table.add_column("Created Files")
        table.add_column("VTK")
        table.add_column("ANBA")
        for r in results:
            section_dir = Path(r["output_dir"])
            vtk_exists = (section_dir / "output.vtk").exists()
            anba_exists = (section_dir / "anba.json").exists()
            table.add_row(
                str(r["section_id"]),
                str(r["success"]),
                "; ".join(r["errors"]),
                r["output_dir"],
                ", ".join([Path(f).name for f in r["created_files"]]),
                "✓" if vtk_exists else "✗",
                "✓" if anba_exists else "✗",
            )
        console.print(table)
        self.logger.info(f"2D meshing completed, outputs in {output_dir}")
