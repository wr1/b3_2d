# b3_2d

A 2D meshing module for b3m, leveraging cgfoil for generating structured meshes from VTP files. This tool processes multi-section airfoil data, extracts ply thicknesses, and produces meshes suitable for finite element analysis.

## Design Overview

b3_2d is designed to handle complex airfoil geometries with multiple sections, skins, and webs. It integrates with pyvista for mesh processing and cgfoil for mesh generation. The architecture separates concerns into CLI, core meshing logic, and plotting utilities, ensuring modularity and ease of maintenance.

Key components:
- **CLI**: Provides command-line interfaces for meshing and plotting via treeparse.
- **Core Meshing**: Processes VTP files, extracts airfoil and web points, defines skins and webs with thickness arrays, and generates meshes using cgfoil.
- **Plotting**: Visualizes meshes with scalar fields for validation.
- **Statesman Integration**: Supports workflow management for b3m pipelines.

The tool uses multiprocessing for parallel section processing, vectorized operations with numpy, and validation with pydantic (where applicable).

## Installation

Install using uv for dependency management:

```bash
uv pip install -e .
```

This installs the package in editable mode, including dependencies like pyvista, cgfoil, numpy, and pydantic.

## Usage

### Meshing

Process a VTP file to generate 2D meshes for each section:

```bash
b3-2d mesh --vtp-file path/to/input.vtp --output-dir path/to/output/
```

Options:
- `--vtp-file` / `-v`: Path to the input VTP file (required).
- `--output-dir` / `-o`: Directory for output files (required).
- `--num-processes` / `-n`: Number of processes for parallel processing (optional, defaults to CPU count).
- `--verbose` / `-V`: Enable verbose logging.

Each section is processed into a subdirectory (e.g., `section_1/`) containing:
- `output.vtk`: VTK mesh file.
- `mesh.pck`: Pickled mesh object.
- `anba.json`: ANBA export for further analysis.
- `2dmesh.log`: Processing log.

### Plotting

Visualize a generated mesh:

```bash
b3-2d plot --mesh-file path/to/mesh.vtk --output-file path/to/plot.png
```

Options:
- `--mesh-file` / `-m`: Path to the input mesh file (required).
- `--output-file` / `-o`: Path to save the plot image (required).
- `--scalar` / `-s`: Scalar field for coloring (default: 'material_id').
- `--verbose` / `-V`: Enable verbose logging.

### Integration with Statesman

For b3m workflows, use the `B32dStep` class in statesman pipelines. It expects a `draped.vtk` from the draping section and outputs to `b3_2d/`.

Example configuration:

```yaml
steps:
  - name: b3_2d
    workdir: ./work
    num_processes: 4
```

## Requirements

- Python 3.8+
- Dependencies: pyvista, cgfoil, numpy, pydantic, rich, treeparse

Ensure VTP files contain 'section_id' and thickness data in cell/point data.
