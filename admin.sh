#!/bin/bash

# Modified files
src/b3_2d/core/mesh.py

ruff format
ruff check --fix > out.txt

git commit src/b3_2d/core/mesh.py -m 'Refactor mesh.py to support variable number of webs instead of hardcoded 2 webs'

uv run pytest -v >> out.txt