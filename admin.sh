#!/bin/bash

ruff format
ruff check --fix > out.txt
uv run pytest -v >> out.txt
