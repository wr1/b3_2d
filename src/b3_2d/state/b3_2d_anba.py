import os
import shutil
import subprocess
from pathlib import Path
from b3_state.core.base import b3_state, ManagedFile
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.console import Console


class b3_2d_anba_step(b3_state):
    """b3_state step for running ANBA4 on 2D meshes."""

    workdir_key = "workdir"
    input_files = [
        ManagedFile(name="b3_2d/", non_empty=True),
    ]
    output_files = ["anba4_results/"]  # Directory with ANBA4 outputs
    dependent_sections = ["b3_2d_mesh"]

    def _execute(self):
        """Execute the step."""
        self.logger.info("Executing B32dAnbaStep: Running ANBA4 on cgfoil outputs.")
        config_dir = Path(self.config_path).parent
        workdir = config_dir / self.config["workdir"]
        output_dir = workdir / "b3_2d"
        anba_files = list(output_dir.glob("section_*/anba.json"))
        if not anba_files:
            self.logger.error("No anba.json files found in b3_2d/")
            return
        anba_env = self.config.get("anba_env", "anba4-env")
        conda_path = os.environ.get("CONDA_EXE") or shutil.which("conda")
        if not conda_path:
            self.logger.error("Conda not found - please install conda")
            return
        result = subprocess.run(
            [conda_path, "env", "list"], capture_output=True, text=True
        )
        if result.returncode != 0 or anba_env not in result.stdout:
            self.logger.error(
                f"Conda environment {anba_env} not found - please create it"
            )
            return
        anba_results_dir = workdir / "anba4_results"
        anba_results_dir.mkdir(exist_ok=True)
        conda_command = [conda_path, "run", "-n", anba_env, "anba4-run", "-i"]
        for anba_file in anba_files:
            conda_command.extend([str(anba_file)])
        self.logger.info(f"Running ANBA4 command: {' '.join(conda_command)}")
        env_vars = {
            **os.environ.copy(),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "-1",
        }
        with Progress(
            SpinnerColumn(), TextColumn("Running ANBA4 on all sections...")
        ) as progress:
            task = progress.add_task("", total=None)
            result = subprocess.run(
                conda_command,
                capture_output=True,
                text=True,
                env=env_vars,
            )
            progress.update(task, completed=True)
        success = result.returncode == 0
        log_file = anba_results_dir / "anba_solve.log"
        with open(log_file, "w") as log:
            log.write("--- ANBA4 run for all sections ---")
            log.write(result.stdout)
            if result.stderr:
                log.write(result.stderr)
        if success:
            self.logger.info(
                f"ANBA4 completed successfully, outputs in {anba_results_dir}"
            )
        else:
            self.logger.info("ANBA4 completed with errors")
        self.logger.info("ANBA4 processing completed")
        # Create table with file existence
        console = Console()
        table = Table(title="Section Processing Results")
        table.add_column("Section ID", justify="right")
        table.add_column("Log")
        table.add_column("VTK")
        table.add_column("ANBA")
        table.add_column("ANBA Out")
        table.add_column("Unit VTU")
        table.add_column("Plot PNG")
        table.add_column("BOM")
        section_dirs = list(output_dir.glob("section_*/"))
        section_dirs.sort(key=lambda d: int(d.name.split("_")[1]))
        for section_dir in section_dirs:
            sid = int(section_dir.name.split("_")[1])
            log_exists = (section_dir / "2dmesh.log").exists()
            vtk_exists = (section_dir / "output.vtk").exists()
            anba_exists = (section_dir / "anba.json").exists()
            anba_out_exists = (section_dir / "anba_out.json").exists()
            unit_vtu_exists = (section_dir / "anba_out_unit.vtu").exists()
            plot_png_exists = (section_dir / "anba_plot.png").exists()
            bom_exists = (section_dir / "bom.json").exists()
            table.add_row(
                str(sid),
                "✓" if log_exists else "✗",
                "✓" if vtk_exists else "✗",
                "✓" if anba_exists else "✗",
                "✓" if anba_out_exists else "✗",
                "✓" if unit_vtu_exists else "✗",
                "✓" if plot_png_exists else "✗",
                "✓" if bom_exists else "✗",
            )
        console.print(table)
