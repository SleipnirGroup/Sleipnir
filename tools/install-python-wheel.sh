#!/bin/bash
set -e

rm -rf .py-build-cmake_cache/
rm -f dist/*.whl

python -m build --wheel
pip install --user --break-system-packages --force-reinstall dist/*.whl
