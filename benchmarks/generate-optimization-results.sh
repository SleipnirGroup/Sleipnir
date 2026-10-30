#!/bin/bash

set -e

for problem in flywheel double_integrator cart_pole; do
  ../build/${problem}_optimization_benchmark \
    --benchmark_min_time=0.05s \
    --benchmark_repetitions=5 \
    --benchmark_report_aggregates_only=true \
    --benchmark_out=${problem//_/-}-optimization-results.csv \
    --benchmark_out_format=csv
done

./plot_results.py optimization \
  --filename flywheel-optimization-results.csv \
  --title Flywheel \
  --noninteractive

./plot_results.py optimization \
  --filename double-integrator-optimization-results.csv \
  --title "Double integrator" \
  --noninteractive

./plot_results.py optimization \
  --filename cart-pole-optimization-results.csv \
  --title Cart-pole \
  --noninteractive
