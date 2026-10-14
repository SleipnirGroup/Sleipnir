#!/bin/bash

../build/flywheel_optimization_benchmark --casadi

echo -n "15 second cooldown..."
sleep 15
echo " done."

../build/flywheel_optimization_benchmark --sleipnir

./plot_optimization_results.py \
  --filenames \
    flywheel-optimization-results-casadi.csv \
    flywheel-optimization-results-sleipnir.csv \
  --labels \
    "CasADi + Ipopt + MUMPS" \
    "Sleipnir" \
  --title Flywheel \
  --noninteractive

echo -n "15 second cooldown..."
sleep 15
echo " done."

../build/cart_pole_optimization_benchmark --casadi

echo -n "30 second cooldown..."
sleep 30
echo " done."

../build/cart_pole_optimization_benchmark --sleipnir

./plot_optimization_results.py \
  --filenames \
    cart-pole-optimization-results-casadi.csv \
    cart-pole-optimization-results-sleipnir.csv \
  --labels \
    "CasADi + Ipopt + MUMPS" \
    "Sleipnir" \
  --title Cart-pole \
  --noninteractive
