#!/bin/bash

# Format code
ruff format

# Check and fix issues, pipe to out.txt
ruff check --fix > out.txt

# Run tests, append to out.txt
uv run pytest -v >> out.txt

# Add and commit new files
git add admin.sh
git commit admin.sh -m 'Add admin.sh script for code formatting, linting, testing, and version control automation'

git add src/b3_2d/core/mesh.py
git commit src/b3_2d/core/mesh.py -m 'Modify process_vtp_multi_section to return section results for display in statesman step'

git add src/b3_2d/state/b3_2d_mesh.py
git commit src/b3_2d/state/b3_2d_mesh.py -m 'Add rich table output to display section processing results in B32dStep'
