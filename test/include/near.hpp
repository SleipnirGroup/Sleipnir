// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>

template <typename T>
bool near(T expected, T actual, T tolerance) {
  using std::abs;
  return abs(expected - actual) < tolerance;
}
