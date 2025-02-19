// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

namespace sleipnir {

/**
 * Expression type.
 *
 * Used for autodiff caching.
 */
enum class ExpressionType : uint8_t {
  /// There is no expression.
  NONE,
  /// The expression is a constant.
  CONSTANT,
  /// The expression is composed of linear and lower-order operators.
  LINEAR,
  /// The expression is composed of quadratic and lower-order operators.
  QUADRATIC,
  /// The expression is composed of nonlinear and lower-order operators.
  NONLINEAR
};

}  // namespace sleipnir
