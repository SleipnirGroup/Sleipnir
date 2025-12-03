// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

/// Sparse matrix coordinate sparsity.
struct Coord {
  int32_t row = 0;
  int32_t col = 0;
  char sign = '0';

  bool operator==(const Coord&) const = default;
};
