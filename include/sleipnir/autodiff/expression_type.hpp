// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <string_view>

#include "sleipnir/util/symbol_exports.hpp"

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

/**
 * Returns user-readable message corresponding to the expression type.
 *
 * @param exit_condition Solver exit condition.
 */
SLEIPNIR_DLLEXPORT constexpr std::string_view ToMessage(
    const ExpressionType& type) {
  using enum ExpressionType;

  switch (type) {
    case NONE:
      return "none";
    case CONSTANT:
      return "constant";
    case LINEAR:
      return "linear";
    case QUADRATIC:
      return "quadratic";
    case NONLINEAR:
      return "nonlinear";
    default:
      return "unknown";
  }
}

}  // namespace sleipnir
