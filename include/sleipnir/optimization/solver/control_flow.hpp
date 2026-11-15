// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

namespace slp {

/// Solver control flow returned by an iteration callback.
enum class ControlFlow : uint8_t {
  /// Continue solving.
  CONTINUE,
  /// Stop solving and return the solution so far.
  BREAK,
};

}  // namespace slp
