// Copyright (c) Sleipnir contributors

#include "Range.hpp"

#include <vector>

std::vector<double> Range(double start, double end, double step) {
  std::vector<double> ret;

  int steps = (end - start) / step;
  for (int i = 0; i < steps; ++i) {
    ret.emplace_back(start + i * step);
  }

  return ret;
}
