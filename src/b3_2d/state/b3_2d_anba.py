import os
import shutil
import subprocess
from pathlib import Path
from statesman import Statesman
from statesman.core.base import ManagedFile
from rich.progress import Progress, SpinnerColumn, TextColumn


class B32dAnbaStep(Statesman):
    """Statesman step for running ANBA4 on 2D meshes."""

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
        env_vars = {
            **os.environ.copy(),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "-1",
        }
        with Progress(SpinnerColumn(), TextColumn("Running ANBA4 on all sections...")) as progress:
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
