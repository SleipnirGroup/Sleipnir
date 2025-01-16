#!/bin/bash

./build/FlywheelScalabilityBenchmark --casadi

echo -n "15 second cooldown..."
sleep 15
echo " done."

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

echo -n "15 second cooldown..."
sleep 15
echo " done."

./build/CartPoleScalabilityBenchmark --casadi

echo -n "30 second cooldown..."
sleep 30
echo " done."

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
