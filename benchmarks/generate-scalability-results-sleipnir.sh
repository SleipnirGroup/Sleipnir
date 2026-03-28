#!/bin/bash
./build/flywheel_scalability_benchmark --sleipnir
./tools/plot_scalability_results.py \
  --filenames \
    flywheel-scalability-results-sleipnir.csv \
  --labels \
    "Sleipnir" \
  --title Flywheel \
  --noninteractive

./build/cart_pole_scalability_benchmark --sleipnir
./tools/plot_scalability_results.py \
  --filenames \
    cart-pole-scalability-results-sleipnir.csv \
  --labels \
    "Sleipnir" \
  --title Cart-pole \
  --noninteractive
