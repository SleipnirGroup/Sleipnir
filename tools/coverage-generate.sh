#!/bin/bash
set -e

if [ $# -eq 0 ]; then
  echo -e "Usage: coverage-generate.sh CMAKE_TARGET\nGenerates coverage of CMake target executable with llvm-cov."
  exit 1
fi

# Build executable
cmake -B build-coverage -S . -DCMAKE_BUILD_TYPE=Coverage -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build-coverage --target $1

# Run executable and generate reports
pushd build-coverage
./$1
llvm-profdata merge -sparse default.profraw -o default.profdata
llvm-cov show -ignore-filename-regex=_deps/ ./$1 -instr-profile=default.profdata -format=html > coverage-line-by-line-$1.html
llvm-cov report -ignore-filename-regex=_deps/ ./$1 -instr-profile=default.profdata > coverage-report-$1.txt
popd
