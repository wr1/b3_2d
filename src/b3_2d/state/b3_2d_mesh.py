from pathlib import Path
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
        process_vtp_multi_section(str(vtp_file), str(output_dir), num_processes, matdb=matdb)
        self.logger.info(f"2D meshing completed, outputs in {output_dir}")
