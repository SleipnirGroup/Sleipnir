// Copyright (c) Sleipnir contributors

#pragma once

#include <iosfwd>

#include "sleipnir/util/SymbolExports.hpp"

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

/**
 * GoogleTest value formatter for ExpressionType.
 *
 * @param type ExpressionType to print.
 * @param os Output stream to which to print.
 */
SLEIPNIR_DLLEXPORT void PrintTo(const ExpressionType& type, std::ostream* os);

}  // namespace sleipnir
