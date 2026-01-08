from pathlib import Path
from rich.table import Table
from rich.console import Console
from statesman import Statesman
from statesman.core.base import ManagedFile
from ..core.mesh import process_vtp_multi_section


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
        results = process_vtp_multi_section(
            str(vtp_file), str(output_dir), num_processes, matdb=matdb
        )
        console = Console()
        table = Table(title="Section Processing Results")
        table.add_column("Section ID", justify="right")
        table.add_column("Success")
        table.add_column("Errors")
        table.add_column("Path")
        table.add_column("Created Files")
        for r in results:
            table.add_row(
                str(r["section_id"]),
                str(r["success"]),
                "; ".join(r["errors"]),
                r["output_dir"],
                ", ".join([Path(f).name for f in r["created_files"]]),
            )
        console.print(table)
        self.logger.info(f"2D meshing completed, outputs in {output_dir}")
