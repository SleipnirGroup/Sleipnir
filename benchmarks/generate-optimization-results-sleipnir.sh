#!/bin/bash
../build/flywheel_optimization_benchmark --sleipnir
./plot_optimization_results.py \
  --filenames \
    flywheel-optimization-results-sleipnir.csv \
  --labels \
    "Sleipnir" \
  --title Flywheel \
  --noninteractive

../build/cart_pole_optimization_benchmark --sleipnir
./plot_optimization_results.py \
  --filenames \
    cart-pole-optimization-results-sleipnir.csv \
  --labels \
    "Sleipnir" \
  --title Cart-pole \
  --noninteractive
