#!/bin/bash
set -e

if [[ $# -ne 1 ]] || [[ "$1" != "cart_pole" && "$1" != "flywheel" ]]; then
  echo "usage: ./tools/perf-benchmark.sh {cart_pole,flywheel}"
  exit 1
fi

cmake -B build-perf -S . -DCMAKE_BUILD_TYPE=Perf -DBUILD_BENCHMARKS=ON -DDISABLE_DIAGNOSTICS=ON
cmake --build build-perf --target $1_perf_benchmark --parallel $(nproc)
./tools/perf-record.sh ./build-perf/$1_perf_benchmark
./tools/perf-report.sh
