#!/bin/bash
# Builds and installs Python wheel for local testing
python -m build --wheel && pip install --user --break-system-packages --force-reinstall dist/*.whl
