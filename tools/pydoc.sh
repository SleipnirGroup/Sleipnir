#!/bin/bash

# Clear workspace
rm -rf .py-build-cmake_cache

# Generate .pyi files
python -m build --wheel --no-isolation

# Copy .pyi files to location for Doxygen to read
cp -r .py-build-cmake_cache/cp312-cp312-linux_x86_64-stubs/_jormungandr jormungandr

# Fix up .pyi contents
find jormungandr -type f -name '*.pyi' -exec sed -i \
  -e 's/sleipnir:://g' \
  -e 's/VariableBlock<VariableMatrix>/VariableBlock/g' \
  -e 's/_jormungandr.autodiff.//g' \
  -e '/import _jormungandr.autodiff/d' \
  -e 's/<ExpressionType.NONE: 0>/ExpressionType.NONE/g' \
  -e 's/<SolverExitCondition.SUCCESS: 0>/SolverExitCondition.SUCCESS/g' \
  {} \;

# Rename .pyi files to .py
find jormungandr -type f -name '*.pyi' |
  while IFS= read filename; do
    mv "$filename" "${filename%.pyi}.py"
  done
