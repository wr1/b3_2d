"""Programmatic example for 2D meshing using b3_2d with 3-web configuration."""

import pyvista as pv
from pathlib import Path
import json
from b3_2d.core.mesh import process_vtp_multi_section
from b3_2d.core.bom import compute_bom
from b3_2d.core.bom_plotting import plot_bom_spanwise
from b3_2d.core.span_plotting import plot_span_anba

# Example: Process draped_3web.vtk for multi-section meshing with 3 webs
vtp_file = "examples/draped_3web.vtk"
output_dir = "output_3web"

# Sample material database (adjust as needed)
matdb = {
    "carbon": {"id": 1, "rho": 1600.0},
    "glass": {"id": 2, "rho": 2000.0},
    "foam": {"id": 3, "rho": 100.0},
}

print(f"Processing 3-web VTP file: {vtp_file}")
results = process_vtp_multi_section(vtp_file, output_dir)

successful = sum(1 for r in results if r["success"])
failed = len(results) - successful
print(f"Processed {len(results)} sections: {successful} successful, {failed} failed.")

if failed > 0:
    print("Failed sections:")
    for r in results:
        if not r["success"]:
            errors_str = "; ".join(r.get("errors", ["Unknown error"]))
            print(f"  Section {r['section_id']}: {errors_str}")

# Detailed summary of created files
print("\nCreated files per section:")
for r in results:
    if r["created_files"]:
        print(f"  Section {r['section_id']} ({'✓' if r['success'] else '✗'}):")
        for f in r["created_files"]:
            print(f"    - {Path(f).name}")

print(f"\nAll outputs saved to: {Path(output_dir).resolve()}")

# Compute and display BOM for successful sections
print("\nBOM calculations:")
for r in results:
    if r["success"]:
        vtk_file = Path(r["output_dir"]) / "output.vtk"
        if vtk_file.exists():
            mesh = pv.read(str(vtk_file))
            bom_data = compute_bom(mesh, matdb)
            if bom_data:
                # Save BOM to JSON
                bom_file = Path(r["output_dir"]) / "bom.json"
                with open(bom_file, "w") as f:
                    json.dump(bom_data, f, indent=2)
                print(f"  Section {r['section_id']}:")
                print(f"    Total Area: {bom_data['total_area']:.2f}")
                print(f"    Areas per Material: {bom_data['areas_per_material']}")
                if "total_mass" in bom_data:
                    print(f"    Total Mass: {bom_data['total_mass']:.2f}")
                    print(f"    Masses per Material: {bom_data['masses_per_material']}")
                print()

# Postprocessing: Plot BOM and ANBA spanwise (test the post step functionality)
print("\nPostprocessing (testing post step):")
bom_plot_file = Path(output_dir) / "bom_spanwise.png"
print(f"Generating BOM spanwise plot: {bom_plot_file}")
plot_bom_spanwise(output_dir, str(bom_plot_file), matdb)
print(f"BOM plot saved to: {bom_plot_file.resolve()}")

anba_plot_file = Path(output_dir) / "anba_spanwise.png"
print(f"Attempting ANBA spanwise plot: {anba_plot_file} (may warn if no ANBA data)")
plot_span_anba(output_dir, str(anba_plot_file))
if anba_plot_file.exists():
    print(f"ANBA plot saved to: {anba_plot_file.resolve()}")
else:
    print("ANBA plot not generated (no ANBA data available)")

# Verify VTK files for successful sections
print("\nVerifying VTK meshes:")
for r in results:
    if r["success"]:
        vtk_file = Path(r["output_dir"]) / "output.vtk"
        if vtk_file.exists():
            mesh = pv.read(str(vtk_file))
            print(
                f"  Section {r['section_id']}: {mesh.n_points} points, {mesh.n_cells} cells"
            )
