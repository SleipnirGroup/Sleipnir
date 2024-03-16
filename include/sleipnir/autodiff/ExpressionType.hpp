// Copyright (c) Sleipnir contributors

#pragma once

namespace sleipnir {

/**
 * Expression type.
 *
 * Used for autodiff caching.
 */
enum class ExpressionType {
  /// There is no expression.
  kNone,
  /// The expression is a constant.
  kConstant,
  /// The expression is composed of linear and lower-order operators.
  kLinear,
  /// The expression is composed of quadratic and lower-order operators.
  kQuadratic,
  /// The expression is composed of nonlinear and lower-order operators.
  kNonlinear
};

}  // namespace sleipnir
