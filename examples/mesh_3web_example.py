"""Programmatic example for 2D meshing using b3_2d with 3-web configuration."""

import pyvista as pv
from pathlib import Path
from b3_2d.core.mesh import process_vtp_multi_section

# Example: Process draped_3web.vtk for multi-section meshing with 3 webs
vtp_file = "examples/draped_3web.vtk"
output_dir = "output_3web"

print(f"Processing 3-web VTP file: {vtp_file}")
results = process_vtp_multi_section(vtp_file, output_dir)

successful = sum(1 for r in results if r['success'])
failed = len(results) - successful
print(f"Processed {len(results)} sections: {successful} successful, {failed} failed.")

if failed > 0:
    print("Failed sections:")
    for r in results:
        if not r['success']:
            errors_str = '; '.join(r.get('errors', ['Unknown error']))
            print(f"  Section {r['section_id']}: {errors_str}")

# Detailed summary of created files
print("\nCreated files per section:")
for r in results:
    if r['created_files']:
        print(f"  Section {r['section_id']} ({'✓' if r['success'] else '✗'}):")
        for f in r['created_files']:
            print(f"    - {Path(f).name}")

print(f"\nAll outputs saved to: {Path(output_dir).resolve()}")

# Verify VTK files for successful sections
print("\nVerifying VTK meshes:")
for r in results:
    if r['success']:
        vtk_file = Path(r['output_dir']) / 'output.vtk'
        if vtk_file.exists():
            mesh = pv.read(str(vtk_file))
            print(f"  Section {r['section_id']}: {mesh.n_points} points, {mesh.n_cells} cells")
