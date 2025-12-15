import os
import shutil
import subprocess
from pathlib import Path
from statesman import Statesman
from statesman.core.base import ManagedFile
import concurrent.futures
import logging


class B32dAnbaStep(Statesman):
    """Statesman step for running ANBA4 on 2D meshes."""

    workdir_key = "workdir"
    input_files = [
        ManagedFile(name="b3_2d/", non_empty=True),
    ]
    output_files = ["anba4_results/"]  # Directory with ANBA4 outputs
    dependent_sections = ["b3_2d_mesh"]

    def run_anba_for_file(self, anba_file, anba_env, conda_path, anba_results_dir):
        """Run ANBA4 for a single anba_file."""
        section_dir = anba_file.parent
        output_file = section_dir / "anba_solve.json"
        log_file = section_dir / "anba_solve.log"
        conda_command = [
            conda_path,
            "run",
            "-n",
            anba_env,
            "anba4-run",
            "-i",
            str(anba_file),
            "-o",
            str(output_file),
        ]
        env_vars = {
            **os.environ.copy(),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "-1",
        }
        result = subprocess.run(
            conda_command,
            capture_output=True,
            text=True,
            env=env_vars,
        )
        success = result.returncode == 0
        with open(log_file, "w") as log:
            log.write(f"--- ANBA4 run for {anba_file} ---\n")
            log.write(result.stdout)
            if result.stderr:
                log.write(result.stderr)
        if success:
            self.logger.info(f"ANBA4 output written to {output_file}")
            # Move output files
            for f in section_dir.glob("*.json"):
                if f != anba_file:
                    shutil.move(str(f), str(anba_results_dir / f.name))
            for f in section_dir.glob("*.vtp"):
                shutil.move(str(f), str(anba_results_dir / f.name))
        return str(anba_file), success

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
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self.run_anba_for_file, anba_file, anba_env, conda_path, anba_results_dir)
                for anba_file in anba_files
            ]
            for future in concurrent.futures.as_completed(futures):
                anba_file_str, success = future.result()
                if success:
                    self.logger.info(f"ANBA4 completed for {anba_file_str}")
                else:
                    self.logger.error(f"ANBA4 failed for {anba_file_str}")
        self.logger.info("ANBA4 processing completed")
