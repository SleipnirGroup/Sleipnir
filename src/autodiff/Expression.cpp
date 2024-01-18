// Copyright (c) Sleipnir contributors

#include "sleipnir/autodiff/Expression.hpp"

#include <cmath>
#include <numbers>

namespace sleipnir::detail {

namespace {
// Instantiate static outside Zero() to avoid atomic initialization check on
// every call to Zero()
static auto kZero = MakeExpressionPtr();
}  // namespace

const ExpressionPtr& Zero() {
  return kZero;
}

Expression::Expression(double value, ExpressionType type)
    : value{value}, type{type} {}

Expression::Expression(ExpressionType type, BinaryFuncDouble valueFunc,
                       TrinaryFuncDouble lhsGradientValueFunc,
                       TrinaryFuncExpr lhsGradientFunc, ExpressionPtr lhs)
    : value{valueFunc(lhs->value, 0.0)},
      type{type},
      valueFunc{valueFunc},
      gradientValueFuncs{lhsGradientValueFunc,
                         [](double, double, double) { return 0.0; }},
      gradientFuncs{lhsGradientFunc,
                    [](const ExpressionPtr&, const ExpressionPtr&,
                       const ExpressionPtr&) { return Zero(); }},
      args{lhs, Zero()} {}

Expression::Expression(ExpressionType type, BinaryFuncDouble valueFunc,
                       TrinaryFuncDouble lhsGradientValueFunc,
                       TrinaryFuncDouble rhsGradientValueFunc,
                       TrinaryFuncExpr lhsGradientFunc,
                       TrinaryFuncExpr rhsGradientFunc, ExpressionPtr lhs,
                       ExpressionPtr rhs)
    : value{valueFunc(lhs->value, rhs->value)},
      type{type},
      valueFunc{valueFunc},
      gradientValueFuncs{lhsGradientValueFunc, rhsGradientValueFunc},
      gradientFuncs{lhsGradientFunc, rhsGradientFunc},
      args{lhs, rhs} {}

bool Expression::IsConstant(double constant) const {
  return type == ExpressionType::kConstant && value == constant;
}

ExpressionPtr operator*(const ExpressionPtr& lhs, const ExpressionPtr& rhs) {
  using enum ExpressionType;

  // Prune expression
  if (lhs->IsConstant(0.0) || rhs->IsConstant(0.0)) {
    return Zero();
  } else if (lhs->IsConstant(1.0)) {
    return rhs;
  } else if (rhs->IsConstant(1.0)) {
    return lhs;
  }

  // Evaluate constant
  if (lhs->type == kConstant && rhs->type == kConstant) {
    return MakeExpressionPtr(lhs->value * rhs->value);
  }

  // Evaluate expression type
  ExpressionType type;
  if (lhs->type == kConstant) {
    type = rhs->type;
  } else if (rhs->type == kConstant) {
    type = lhs->type;
  } else if (lhs->type == kLinear && rhs->type == kLinear) {
    type = ExpressionType::kQuadratic;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return MakeExpressionPtr(
      type, [](double lhs, double rhs) { return lhs * rhs; },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint * rhs;
      },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint * lhs;
      },
      [](const ExpressionPtr& lhs, const ExpressionPtr& rhs,
         const ExpressionPtr& parentAdjoint) { return parentAdjoint * rhs; },
      [](const ExpressionPtr& lhs, const ExpressionPtr& rhs,
         const ExpressionPtr& parentAdjoint) { return parentAdjoint * lhs; },
      lhs, rhs);
}

ExpressionPtr operator/(const ExpressionPtr& lhs, const ExpressionPtr& rhs) {
  using enum ExpressionType;

  // Prune expression
  if (lhs->IsConstant(0.0)) {
    return Zero();
  } else if (rhs->IsConstant(1.0)) {
    return lhs;
  }

  // Evaluate constant
  if (lhs->type == kConstant && rhs->type == kConstant) {
    return MakeExpressionPtr(lhs->value / rhs->value);
  }

  // Evaluate expression type
  ExpressionType type;
  if (rhs->type == kConstant) {
    type = lhs->type;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return MakeExpressionPtr(
      type, [](double lhs, double rhs) { return lhs / rhs; },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint / rhs;
      },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint * -lhs / (rhs * rhs);
      },
      [](const ExpressionPtr& lhs, const ExpressionPtr& rhs,
         const ExpressionPtr& parentAdjoint) { return parentAdjoint / rhs; },
      [](const ExpressionPtr& lhs, const ExpressionPtr& rhs,
         const ExpressionPtr& parentAdjoint) {
        return parentAdjoint * -lhs / (rhs * rhs);
      },
      lhs, rhs);
}

ExpressionPtr operator+(const ExpressionPtr& lhs, const ExpressionPtr& rhs) {
  using enum ExpressionType;

  // Prune expression
  if (lhs->IsConstant(0.0)) {
    return rhs;
  } else if (rhs->IsConstant(0.0)) {
    return lhs;
  }

  // Evaluate constant
  if (lhs->type == kConstant && rhs->type == kConstant) {
    return MakeExpressionPtr(lhs->value + rhs->value);
  }

  return MakeExpressionPtr(
      std::max(lhs->type, rhs->type),
      [](double lhs, double rhs) { return lhs + rhs; },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint;
      },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint;
      },
      [](const ExpressionPtr& lhs, const ExpressionPtr& rhs,
         const ExpressionPtr& parentAdjoint) { return parentAdjoint; },
      [](const ExpressionPtr& lhs, const ExpressionPtr& rhs,
         const ExpressionPtr& parentAdjoint) { return parentAdjoint; },
      lhs, rhs);
}

ExpressionPtr operator-(const ExpressionPtr& lhs, const ExpressionPtr& rhs) {
  using enum ExpressionType;

  // Prune expression
  if (lhs->IsConstant(0.0)) {
    if (rhs->IsConstant(0.0)) {
      return Zero();
    } else {
      return -rhs;
    }
  } else if (rhs->IsConstant(0.0)) {
    return lhs;
  }

  // Evaluate constant
  if (lhs->type == kConstant && rhs->type == kConstant) {
    return MakeExpressionPtr(lhs->value - rhs->value);
  }

  return MakeExpressionPtr(
      std::max(lhs->type, rhs->type),
      [](double lhs, double rhs) { return lhs - rhs; },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint;
      },
      [](double lhs, double rhs, double parentAdjoint) {
        return -parentAdjoint;
      },
      [](const ExpressionPtr& lhs, const ExpressionPtr& rhs,
         const ExpressionPtr& parentAdjoint) { return parentAdjoint; },
      [](const ExpressionPtr& lhs, const ExpressionPtr& rhs,
         const ExpressionPtr& parentAdjoint) { return -parentAdjoint; },
      lhs, rhs);
}

ExpressionPtr operator-(const ExpressionPtr& lhs) {
  using enum ExpressionType;

  // Prune expression
  if (lhs->IsConstant(0.0)) {
    return Zero();
  }

  // Evaluate constant
  if (lhs->type == kConstant) {
    return MakeExpressionPtr(-lhs->value);
  }

  return MakeExpressionPtr(
      lhs->type, [](double lhs, double) { return -lhs; },
      [](double lhs, double, double parentAdjoint) { return -parentAdjoint; },
      [](const ExpressionPtr& lhs, const ExpressionPtr& rhs,
         const ExpressionPtr& parentAdjoint) { return -parentAdjoint; },
      lhs);
}

ExpressionPtr operator+(const ExpressionPtr& lhs) {
  // Prune expression
  if (lhs->IsConstant(0.0)) {
    return Zero();
  } else {
    return lhs;
  }
}

ExpressionPtr abs(const ExpressionPtr& x) {  // NOLINT
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return Zero();
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr(std::abs(x->value));
  }

  return MakeExpressionPtr(
      kNonlinear, [](double x, double) { return std::abs(x); },
      [](double x, double, double parentAdjoint) {
        if (x < 0.0) {
          return -parentAdjoint;
        } else if (x > 0.0) {
          return parentAdjoint;
        } else {
          return 0.0;
        }
      },
      [](const ExpressionPtr& x, const ExpressionPtr&,
         const ExpressionPtr& parentAdjoint) {
        if (x->value < 0.0) {
          return -parentAdjoint;
        } else if (x->value > 0.0) {
          return parentAdjoint;
        } else {
          return Zero();
        }
      },
      x);
}

ExpressionPtr acos(const ExpressionPtr& x) {  // NOLINT
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return MakeExpressionPtr(std::numbers::pi / 2.0);
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr(std::acos(x->value));
  }

  return MakeExpressionPtr(
      kNonlinear, [](double x, double) { return std::acos(x); },
      [](double x, double, double parentAdjoint) {
        return -parentAdjoint / std::sqrt(1.0 - x * x);
      },
      [](const ExpressionPtr& x, const ExpressionPtr&,
         const ExpressionPtr& parentAdjoint) {
        return -parentAdjoint /
               sleipnir::detail::sqrt(MakeExpressionPtr(1.0) - x * x);
      },
      x);
}

ExpressionPtr asin(  // NOLINT
    const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return Zero();
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr(std::asin(x->value));
  }

  return MakeExpressionPtr(
      kNonlinear, [](double x, double) { return std::asin(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / std::sqrt(1.0 - x * x);
      },
      [](const ExpressionPtr& x, const ExpressionPtr&,
         const ExpressionPtr& parentAdjoint) {
        return parentAdjoint /
               sleipnir::detail::sqrt(MakeExpressionPtr(1.0) - x * x);
      },
      x);
}

ExpressionPtr atan(const ExpressionPtr& x) {  // NOLINT
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return Zero();
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr(std::atan(x->value));
  }

  return MakeExpressionPtr(
      kNonlinear, [](double x, double) { return std::atan(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / (1.0 + x * x);
      },
      [](const ExpressionPtr& x, const ExpressionPtr&,
         const ExpressionPtr& parentAdjoint) {
        return parentAdjoint / (MakeExpressionPtr(1.0) + x * x);
      },
      x);
}

ExpressionPtr atan2(const ExpressionPtr& y, const ExpressionPtr& x) {  // NOLINT
  using enum ExpressionType;

  // Prune expression
  if (y->IsConstant(0.0)) {
    return Zero();
  } else if (x->IsConstant(0.0)) {
    return MakeExpressionPtr(std::numbers::pi / 2.0);
  }

  // Evaluate constant
  if (y->type == kConstant && x->type == kConstant) {
    return MakeExpressionPtr(std::atan2(y->value, x->value));
  }

  return MakeExpressionPtr(
      kNonlinear, [](double y, double x) { return std::atan2(y, x); },
      [](double y, double x, double parentAdjoint) {
        return parentAdjoint * x / (y * y + x * x);
      },
      [](double y, double x, double parentAdjoint) {
        return parentAdjoint * -y / (y * y + x * x);
      },
      [](const ExpressionPtr& y, const ExpressionPtr& x,
         const ExpressionPtr& parentAdjoint) {
        return parentAdjoint * x / (y * y + x * x);
      },
      [](const ExpressionPtr& y, const ExpressionPtr& x,
         const ExpressionPtr& parentAdjoint) {
        return parentAdjoint * -y / (y * y + x * x);
      },
      y, x);
}

ExpressionPtr cos(const ExpressionPtr& x) {  // NOLINT
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return MakeExpressionPtr(1.0);
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr(std::cos(x->value));
  }

  return MakeExpressionPtr(
      kNonlinear, [](double x, double) { return std::cos(x); },
      [](double x, double, double parentAdjoint) {
        return -parentAdjoint * std::sin(x);
      },
      [](const ExpressionPtr& x, const ExpressionPtr&,
         const ExpressionPtr& parentAdjoint) {
        return parentAdjoint * -sleipnir::detail::sin(x);
      },
      x);
}

ExpressionPtr cosh(const ExpressionPtr& x) {  // NOLINT
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return MakeExpressionPtr(1.0);
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr(std::cosh(x->value));
  }

  return MakeExpressionPtr(
      kNonlinear, [](double x, double) { return std::cosh(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint * std::sinh(x);
      },
      [](const ExpressionPtr& x, const ExpressionPtr&,
         const ExpressionPtr& parentAdjoint) {
        return parentAdjoint * sleipnir::detail::sinh(x);
      },
      x);
}

ExpressionPtr erf(const ExpressionPtr& x) {  // NOLINT
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return Zero();
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr(std::erf(x->value));
  }

  return MakeExpressionPtr(
      kNonlinear, [](double x, double) { return std::erf(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint * 2.0 * std::numbers::inv_sqrtpi *
               std::exp(-x * x);
      },
      [](const ExpressionPtr& x, const ExpressionPtr&,
         const ExpressionPtr& parentAdjoint) {
        return parentAdjoint *
               MakeExpressionPtr(2.0 * std::numbers::inv_sqrtpi) *
               sleipnir::detail::exp(-x * x);
      },
      x);
}

ExpressionPtr exp(const ExpressionPtr& x) {  // NOLINT
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return MakeExpressionPtr(1.0);
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr(std::exp(x->value));
  }

  return MakeExpressionPtr(
      kNonlinear, [](double x, double) { return std::exp(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint * std::exp(x);
      },
      [](const ExpressionPtr& x, const ExpressionPtr&,
         const ExpressionPtr& parentAdjoint) {
        return parentAdjoint * sleipnir::detail::exp(x);
      },
      x);
}

ExpressionPtr hypot(const ExpressionPtr& x, const ExpressionPtr& y) {  // NOLINT
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0) && y->IsConstant(0.0)) {
    return Zero();
  } else if (x->IsConstant(0.0)) {
    return y;
  } else if (y->IsConstant(0.0)) {
    return x;
  }

  // Evaluate constant
  if (x->type == kConstant && y->type == kConstant) {
    return MakeExpressionPtr(std::hypot(x->value, y->value));
  }

  // Evaluate expression type
  ExpressionType type;
  if (x->IsConstant(0.0)) {
    type = y->type;
  } else if (y->IsConstant(0.0)) {
    type = x->type;
  } else {
    type = kNonlinear;
  }

  return MakeExpressionPtr(
      type, [](double x, double y) { return std::hypot(x, y); },
      [](double x, double y, double parentAdjoint) {
        return parentAdjoint * x / std::hypot(x, y);
      },
      [](double x, double y, double parentAdjoint) {
        return parentAdjoint * y / std::hypot(x, y);
      },
      [](const ExpressionPtr& x, const ExpressionPtr& y,
         const ExpressionPtr& parentAdjoint) {
        return parentAdjoint * x / sleipnir::detail::hypot(x, y);
      },
      [](const ExpressionPtr& x, const ExpressionPtr& y,
         const ExpressionPtr& parentAdjoint) {
        return parentAdjoint * y / sleipnir::detail::hypot(x, y);
      },
      x, y);
}

ExpressionPtr log(const ExpressionPtr& x) {  // NOLINT
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return Zero();
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr(std::log(x->value));
  }

  return MakeExpressionPtr(
      kNonlinear, [](double x, double) { return std::log(x); },
      [](double x, double, double parentAdjoint) { return parentAdjoint / x; },
      [](const ExpressionPtr& x, const ExpressionPtr&,
         const ExpressionPtr& parentAdjoint) { return parentAdjoint / x; },
      x);
}

ExpressionPtr log10(const ExpressionPtr& x) {  // NOLINT
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return Zero();
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr(std::log10(x->value));
  }

  return MakeExpressionPtr(
      kNonlinear, [](double x, double) { return std::log10(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / (std::numbers::ln10 * x);
      },
      [](const ExpressionPtr& x, const ExpressionPtr&,
         const ExpressionPtr& parentAdjoint) {
        return parentAdjoint / (MakeExpressionPtr(std::numbers::ln10) * x);
      },
      x);
}

ExpressionPtr pow(const ExpressionPtr& base,  // NOLINT
                  const ExpressionPtr& power) {
  using enum ExpressionType;

  // Prune expression
  if (base->IsConstant(0.0)) {
    return Zero();
  } else if (base->IsConstant(1.0)) {
    return base;
  }
  if (power->IsConstant(0.0)) {
    return MakeExpressionPtr(1.0);
  } else if (power->IsConstant(1.0)) {
    return base;
  }

  // Evaluate constant
  if (base->type == kConstant && power->type == kConstant) {
    return MakeExpressionPtr(std::pow(base->value, power->value));
  }

  return MakeExpressionPtr(
      base->type == kLinear && power->IsConstant(2.0) ? kQuadratic : kNonlinear,
      [](double base, double power) { return std::pow(base, power); },
      [](double base, double power, double parentAdjoint) {
        return parentAdjoint * std::pow(base, power - 1) * power;
      },
      [](double base, double power, double parentAdjoint) {
        // Since x * std::log(x) -> 0 as x -> 0
        if (base == 0.0) {
          return 0.0;
        } else {
          return parentAdjoint * std::pow(base, power - 1) * base *
                 std::log(base);
        }
      },
      [](const ExpressionPtr& base, const ExpressionPtr& power,
         const ExpressionPtr& parentAdjoint) {
        return parentAdjoint *
               sleipnir::detail::pow(base, power - MakeExpressionPtr(1.0)) *
               power;
      },
      [](const ExpressionPtr& base, const ExpressionPtr& power,
         const ExpressionPtr& parentAdjoint) {
        // Since x * std::log(x) -> 0 as x -> 0
        if (base->value == 0.0) {
          return Zero();
        } else {
          return parentAdjoint *
                 sleipnir::detail::pow(base, power - MakeExpressionPtr(1.0)) *
                 base * sleipnir::detail::log(base);
        }
      },
      base, power);
}

ExpressionPtr sign(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return Zero();
  }

  // Evaluate constant
  if (x->type == kConstant) {
    if (x->value < 0.0) {
      return MakeExpressionPtr(-1.0);
    } else if (x->value == 0.0) {
      return MakeExpressionPtr(0.0);
    } else {
      return MakeExpressionPtr(1.0);
    }
  }

  return MakeExpressionPtr(
      kNonlinear,
      [](double x, double) {
        if (x < 0.0) {
          return -1.0;
        } else if (x == 0.0) {
          return 0.0;
        } else {
          return 1.0;
        }
      },
      [](double x, double, double parentAdjoint) { return 0.0; },
      [](const ExpressionPtr& x, const ExpressionPtr&,
         const ExpressionPtr& parentAdjoint) { return Zero(); },
      x);
}

ExpressionPtr sin(const ExpressionPtr& x) {  // NOLINT
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return Zero();
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr(std::sin(x->value));
  }

  return MakeExpressionPtr(
      kNonlinear, [](double x, double) { return std::sin(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint * std::cos(x);
      },
      [](const ExpressionPtr& x, const ExpressionPtr&,
         const ExpressionPtr& parentAdjoint) {
        return parentAdjoint * sleipnir::detail::cos(x);
      },
      x);
}

ExpressionPtr sinh(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return Zero();
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr(std::sinh(x->value));
  }

  return MakeExpressionPtr(
      kNonlinear, [](double x, double) { return std::sinh(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint * std::cosh(x);
      },
      [](const ExpressionPtr& x, const ExpressionPtr&,
         const ExpressionPtr& parentAdjoint) {
        return parentAdjoint * sleipnir::detail::cosh(x);
      },
      x);
}

ExpressionPtr sqrt(const ExpressionPtr& x) {  // NOLINT
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return Zero();
  } else if (x->IsConstant(1.0)) {
    return x;
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr(std::sqrt(x->value));
  }

  return MakeExpressionPtr(
      kNonlinear, [](double x, double) { return std::sqrt(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / (2.0 * std::sqrt(x));
      },
      [](const ExpressionPtr& x, const ExpressionPtr&,
         const ExpressionPtr& parentAdjoint) {
        return parentAdjoint /
               (MakeExpressionPtr(2.0) * sleipnir::detail::sqrt(x));
      },
      x);
}

ExpressionPtr tan(const ExpressionPtr& x) {  // NOLINT
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return Zero();
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr(std::tan(x->value));
  }

  return MakeExpressionPtr(
      kNonlinear, [](double x, double) { return std::tan(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / (std::cos(x) * std::cos(x));
      },
      [](const ExpressionPtr& x, const ExpressionPtr&,
         const ExpressionPtr& parentAdjoint) {
        return parentAdjoint /
               (sleipnir::detail::cos(x) * sleipnir::detail::cos(x));
      },
      x);
}

ExpressionPtr tanh(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return Zero();
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr(std::tanh(x->value));
  }

  return MakeExpressionPtr(
      kNonlinear, [](double x, double) { return std::tanh(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / (std::cosh(x) * std::cosh(x));
      },
      [](const ExpressionPtr& x, const ExpressionPtr&,
         const ExpressionPtr& parentAdjoint) {
        return parentAdjoint /
               (sleipnir::detail::cosh(x) * sleipnir::detail::cosh(x));
      },
      x);
}

}  // namespace sleipnir::detail

namespace sleipnir {

// Instantiate Expression pool in Expression.cpp instead to avoid ODR violation
template EXPORT_TEMPLATE_DEFINE(SLEIPNIR_DLLEXPORT)
    PoolAllocator<detail::Expression> GlobalPoolAllocator<detail::Expression>();

}  // namespace sleipnir
