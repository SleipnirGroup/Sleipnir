#!/bin/bash
./build/FlywheelScalabilityBenchmark --casadi
./build/FlywheelScalabilityBenchmark --sleipnir
./tools/plot_scalability_results.py \
  --filenames \
    flywheel-scalability-results-casadi.csv \
    flywheel-scalability-results-sleipnir.csv \
  --labels \
    "CasADi + Ipopt + MUMPS" \
    "Sleipnir" \
  --title Flywheel \
  --noninteractive

./build/CartPoleScalabilityBenchmark --casadi
./build/CartPoleScalabilityBenchmark --sleipnir
./tools/plot_scalability_results.py \
  --filenames \
    cart-pole-scalability-results-casadi.csv \
    cart-pole-scalability-results-sleipnir.csv \
  --labels \
    "CasADi + Ipopt + MUMPS" \
    "Sleipnir" \
  --title Cart-pole \
  --noninteractive
