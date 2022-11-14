#!/bin/bash
./build/FlywheelScalabilityBenchmark
./tools/plot_scalability_results.py --filename flywheel-scalability-results.csv --title Flywheel --noninteractive

./build/CartPoleScalabilityBenchmark
./tools/plot_scalability_results.py --filename cart-pole-scalability-results.csv --title Cart-pole --noninteractive
