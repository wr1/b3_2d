from pathlib import Path
from statesman import Statesman
from statesman.core.base import ManagedFile
from ..core.bom_plotting import plot_bom_spanwise
from ..core.span_plotting import plot_span_anba


class B32dPostStep(Statesman):
    """Statesman step for postprocessing 2D meshes and ANBA results."""

    workdir_key = "workdir"
    input_files = [
        ManagedFile(name="b3_2d/", non_empty=True),
        # ANBA results are optional
    ]
    output_files = ["b3_2d/bom_spanwise.png", "b3_2d/anba_spanwise.png"]
    dependent_sections = ["b3_2d_mesh"]  # ANBA is optional

    def _execute(self):
        """Execute the postprocessing step."""
        self.logger.info("Executing B32dPostStep: Postprocessing 2D meshes and ANBA results.")
        config_dir = Path(self.config_path).parent
        workdir = config_dir / self.config["workdir"]
        output_dir = workdir / "b3_2d"
        matdb = self.config.get("matdb", {})

        # Plot BOM spanwise if data exists
        bom_plot_file = output_dir / "bom_spanwise.png"
        self.logger.info(f"Attempting to generate BOM spanwise plot: {bom_plot_file}")
        plot_bom_spanwise(str(output_dir), str(bom_plot_file), matdb)
        if bom_plot_file.exists():
            self.logger.info(f"BOM plot saved to {bom_plot_file}")
        else:
            self.logger.warning(f"BOM plot not generated (no BOM data found): {bom_plot_file}")

        # Plot ANBA spanwise if data exists
        anba_plot_file = output_dir / "anba_spanwise.png"
        self.logger.info(f"Attempting to generate ANBA spanwise plot: {anba_plot_file}")
        plot_span_anba(str(output_dir), str(anba_plot_file))
        if anba_plot_file.exists():
            self.logger.info(f"ANBA plot saved to {anba_plot_file}")
        else:
            self.logger.warning(f"ANBA plot not generated (no ANBA data found): {anba_plot_file}")

        self.logger.info("Postprocessing completed.")
