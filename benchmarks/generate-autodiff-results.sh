#!/bin/bash

for derivative in gradient jacobian hessian; do
  ../build/${derivative}_autodiff_benchmark \
    --benchmark_min_time=0.05s \
    --benchmark_repetitions=5 \
    --benchmark_report_aggregates_only=true \
    --benchmark_out=${derivative}-autodiff-results.csv \
    --benchmark_out_format=csv
done

for derivative in Gradient Jacobian Hessian; do
  ./plot_results.py autodiff \
    --filename ${derivative,,}-autodiff-results.csv \
    --title $derivative \
    --noninteractive
done
