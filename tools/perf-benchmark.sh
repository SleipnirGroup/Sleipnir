#!/bin/bash
set -e

if [[ $# -ne 1 ]] || [[ "$1" != "CartPole" && "$1" != "Flywheel" ]]; then
  echo "usage: ./tools/perf-benchmark.sh {CartPole,Flywheel}"
  exit 1
fi

cmake -B build-perf -S . -DCMAKE_BUILD_TYPE=Perf -DBUILD_BENCHMARKS=ON -DDISABLE_DIAGNOSTICS=ON
cmake --build build-perf --target $1PerfBenchmark
./tools/perf-record.sh ./build-perf/$1PerfBenchmark
./tools/perf-report.sh
