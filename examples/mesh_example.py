"""Programmatic example for 2D meshing using b3_2d."""

from b3_2d.core.mesh import process_vtp_multi_section

# Example: Process a VTP file for multi-section meshing
vtp_file = "examples/draped.vtk"
output_dir = "output"

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
