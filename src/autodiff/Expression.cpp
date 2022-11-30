// Copyright (c) Joshua Nichols and Tyler Veness

#include "sleipnir/autodiff/Expression.hpp"

#include <cmath>
#include <numbers>
#include <type_traits>

#include "sleipnir/autodiff/ExpressionGraph.hpp"

// https://en.cppreference.com/w/cpp/utility/to_underlying from C++23
template <class Enum>
constexpr std::underlying_type_t<Enum> to_underlying(Enum e) noexcept {
  return static_cast<std::underlying_type_t<Enum>>(e);
}

namespace sleipnir {

IntrusiveSharedPtr<Expression>& Zero() {
  static auto expr = AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), Expression::ZeroSingleton);
  return expr;
}

Expression::Expression(ZeroSingleton_t)
    : adjointExpr{nullptr},
      type{ExpressionType::kConstant},
      args{nullptr, nullptr} {}

Expression::Expression(double value, ExpressionType type)
    : value{value}, adjointExpr{Zero()}, type{type}, args{Zero(), Zero()} {}

Expression::Expression(ExpressionType type, BinaryFuncDouble valueFunc,
                       TrinaryFuncDouble lhsGradientValueFunc,
                       TrinaryFuncExpr lhsGradientFunc,
                       IntrusiveSharedPtr<Expression> lhs)
    : value{valueFunc(lhs->value, 0.0)},
      adjointExpr{Zero()},
      type{type},
      valueFunc{valueFunc},
      gradientValueFuncs{lhsGradientValueFunc,
                         [](double, double, double) { return 0.0; }},
      gradientFuncs{
          lhsGradientFunc,
          [](const IntrusiveSharedPtr<Expression>&,
             const IntrusiveSharedPtr<Expression>&,
             const IntrusiveSharedPtr<Expression>&) { return Zero(); }},
      args{lhs, Zero()} {}

Expression::Expression(ExpressionType type, BinaryFuncDouble valueFunc,
                       TrinaryFuncDouble lhsGradientValueFunc,
                       TrinaryFuncDouble rhsGradientValueFunc,
                       TrinaryFuncExpr lhsGradientFunc,
                       TrinaryFuncExpr rhsGradientFunc,
                       IntrusiveSharedPtr<Expression> lhs,
                       IntrusiveSharedPtr<Expression> rhs)
    : value{valueFunc(lhs->value, rhs->value)},
      adjointExpr{Zero()},
      type{type},
      valueFunc{valueFunc},
      gradientValueFuncs{lhsGradientValueFunc, rhsGradientValueFunc},
      gradientFuncs{lhsGradientFunc, rhsGradientFunc},
      args{lhs, rhs} {}

SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator*(
    double lhs, const IntrusiveSharedPtr<Expression>& rhs) {
  if (lhs == 0.0) {
    return Zero();
  } else if (lhs == 1.0) {
    return rhs;
  }

  return MakeConstant(lhs) * rhs;
}

SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator*(
    const IntrusiveSharedPtr<Expression>& lhs, double rhs) {
  if (rhs == 0.0) {
    return Zero();
  } else if (rhs == 1.0) {
    return lhs;
  }

  return lhs * MakeConstant(rhs);
}

SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator*(
    const IntrusiveSharedPtr<Expression>& lhs,
    const IntrusiveSharedPtr<Expression>& rhs) {
  if (lhs == Zero() || rhs == Zero()) {
    return Zero();
  }

  if (lhs->type == ExpressionType::kConstant) {
    if (lhs->value == 1.0) {
      return rhs;
    } else if (lhs->value == 0.0) {
      return Zero();
    }
  }

  if (rhs->type == ExpressionType::kConstant) {
    if (rhs->value == 1.0) {
      return lhs;
    } else if (rhs->value == 0.0) {
      return Zero();
    }
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (lhs->type == ExpressionType::kConstant) {
    type = rhs->type;
  } else if (rhs->type == ExpressionType::kConstant) {
    type = lhs->type;
  } else if (lhs->type == ExpressionType::kLinear &&
             rhs->type == ExpressionType::kLinear) {
    type = ExpressionType::kQuadratic;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double lhs, double rhs) { return lhs * rhs; },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint * rhs;
      },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint * lhs;
      },
      [](const IntrusiveSharedPtr<Expression>& lhs,
         const IntrusiveSharedPtr<Expression>& rhs,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * rhs;
      },
      [](const IntrusiveSharedPtr<Expression>& lhs,
         const IntrusiveSharedPtr<Expression>& rhs,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * lhs;
      },
      lhs, rhs);
}

SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator/(
    double lhs, const IntrusiveSharedPtr<Expression>& rhs) {
  if (lhs == 0.0) {
    return Zero();
  }

  return MakeConstant(lhs) / rhs;
}

SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator/(
    const IntrusiveSharedPtr<Expression>& lhs, double rhs) {
  return lhs / MakeConstant(rhs);
}

SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator/(
    const IntrusiveSharedPtr<Expression>& lhs,
    const IntrusiveSharedPtr<Expression>& rhs) {
  if (lhs == Zero()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (rhs->type == ExpressionType::kConstant) {
    type = lhs->type;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double lhs, double rhs) { return lhs / rhs; },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint / rhs;
      },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint * -lhs / (rhs * rhs);
      },
      [](const IntrusiveSharedPtr<Expression>& lhs,
         const IntrusiveSharedPtr<Expression>& rhs,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint / rhs;
      },
      [](const IntrusiveSharedPtr<Expression>& lhs,
         const IntrusiveSharedPtr<Expression>& rhs,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * -lhs / (rhs * rhs);
      },
      lhs, rhs);
}

SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator+(
    double lhs, const IntrusiveSharedPtr<Expression>& rhs) {
  if (lhs == 0.0) {
    return rhs;
  }

  return MakeConstant(lhs) + rhs;
}

SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator+(
    const IntrusiveSharedPtr<Expression>& lhs, double rhs) {
  if (rhs == 0.0) {
    return lhs;
  }

  return lhs + MakeConstant(rhs);
}

SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator+(
    const IntrusiveSharedPtr<Expression>& lhs,
    const IntrusiveSharedPtr<Expression>& rhs) {
  if (lhs == Zero()) {
    return rhs;
  } else if (rhs == Zero()) {
    return lhs;
  }

  return AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(),
      ExpressionType{
          std::max(to_underlying(lhs->type), to_underlying(rhs->type))},
      [](double lhs, double rhs) { return lhs + rhs; },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint;
      },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint;
      },
      [](const IntrusiveSharedPtr<Expression>& lhs,
         const IntrusiveSharedPtr<Expression>& rhs,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint;
      },
      [](const IntrusiveSharedPtr<Expression>& lhs,
         const IntrusiveSharedPtr<Expression>& rhs,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint;
      },
      lhs, rhs);
}

SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression>& operator+=(
    IntrusiveSharedPtr<Expression>& lhs, double rhs) {
  if (rhs == 0.0) {
    return lhs;
  }

  return lhs += MakeConstant(rhs);
}

SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression>& operator+=(
    IntrusiveSharedPtr<Expression>& lhs,
    const IntrusiveSharedPtr<Expression>& rhs) {
  if (lhs == Zero()) {
    lhs = rhs;
  } else if (rhs == Zero()) {
    return lhs;
  } else {
    lhs = lhs + rhs;
  }

  return lhs;
}

SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator-(
    double lhs, const IntrusiveSharedPtr<Expression>& rhs) {
  if (lhs == 0.0) {
    return -rhs;
  }

  return MakeConstant(lhs) - rhs;
}

SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator-(
    const IntrusiveSharedPtr<Expression>& lhs, double rhs) {
  if (rhs == 0.0) {
    return lhs;
  }

  return lhs - MakeConstant(rhs);
}

SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator-(
    const IntrusiveSharedPtr<Expression>& lhs,
    const IntrusiveSharedPtr<Expression>& rhs) {
  if (lhs == Zero()) {
    if (rhs != Zero()) {
      return -rhs;
    } else {
      return Zero();
    }
  } else if (rhs == Zero()) {
    return lhs;
  }

  return AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(),
      ExpressionType{
          std::max(to_underlying(lhs->type), to_underlying(rhs->type))},
      [](double lhs, double rhs) { return lhs - rhs; },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint;
      },
      [](double lhs, double rhs, double parentAdjoint) {
        return -parentAdjoint;
      },
      [](const IntrusiveSharedPtr<Expression>& lhs,
         const IntrusiveSharedPtr<Expression>& rhs,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint;
      },
      [](const IntrusiveSharedPtr<Expression>& lhs,
         const IntrusiveSharedPtr<Expression>& rhs,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return -parentAdjoint;
      },
      lhs, rhs);
}

SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator-(
    const IntrusiveSharedPtr<Expression>& lhs) {
  if (lhs == Zero()) {
    return Zero();
  }

  return AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), lhs->type,
      [](double lhs, double) { return -lhs; },
      [](double lhs, double, double parentAdjoint) { return -parentAdjoint; },
      [](const IntrusiveSharedPtr<Expression>& lhs,
         const IntrusiveSharedPtr<Expression>& rhs,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return -parentAdjoint;
      },
      lhs);
}

SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator+(
    const IntrusiveSharedPtr<Expression>& lhs) {
  if (lhs == Zero()) {
    return Zero();
  }

  return AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), lhs->type,
      [](double lhs, double) { return lhs; },
      [](double lhs, double, double parentAdjoint) { return parentAdjoint; },
      [](const IntrusiveSharedPtr<Expression>& lhs,
         const IntrusiveSharedPtr<Expression>& rhs,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint;
      },
      lhs);
}

IntrusiveSharedPtr<Expression> MakeConstant(double x) {
  return AllocateIntrusiveShared<Expression>(GlobalPoolAllocator<Expression>(),
                                             x, ExpressionType::kConstant);
}

IntrusiveSharedPtr<Expression> abs(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x) {
  if (x == Zero()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::abs(x); },
      [](double x, double, double parentAdjoint) {
        if (x < 0.0) {
          return -parentAdjoint;
        } else if (x > 0.0) {
          return parentAdjoint;
        } else {
          return 0.0;
        }
      },
      [](const IntrusiveSharedPtr<Expression>& x,
         const IntrusiveSharedPtr<Expression>&,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
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

IntrusiveSharedPtr<Expression> acos(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x) {
  if (x == Zero()) {
    return MakeConstant(std::numbers::pi / 2.0);
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::acos(x); },
      [](double x, double, double parentAdjoint) {
        return -parentAdjoint / std::sqrt(1.0 - x * x);
      },
      [](const IntrusiveSharedPtr<Expression>& x,
         const IntrusiveSharedPtr<Expression>&,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return -parentAdjoint / sleipnir::sqrt(1.0 - x * x);
      },
      x);
}

IntrusiveSharedPtr<Expression> asin(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x) {
  if (x == Zero()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::asin(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / std::sqrt(1.0 - x * x);
      },
      [](const IntrusiveSharedPtr<Expression>& x,
         const IntrusiveSharedPtr<Expression>&,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint / sleipnir::sqrt(1.0 - x * x);
      },
      x);
}

IntrusiveSharedPtr<Expression> atan(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x) {
  if (x == Zero()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::atan(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / (1.0 + x * x);
      },
      [](const IntrusiveSharedPtr<Expression>& x,
         const IntrusiveSharedPtr<Expression>&,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint / (1.0 + x * x);
      },
      x);
}

IntrusiveSharedPtr<Expression> atan2(  // NOLINT
    const IntrusiveSharedPtr<Expression>& y,
    const IntrusiveSharedPtr<Expression>& x) {
  if (y == Zero()) {
    return Zero();
  } else if (x == Zero()) {
    return MakeConstant(std::numbers::pi / 2.0);
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (y->type == ExpressionType::kConstant &&
      x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double y, double x) { return std::atan2(y, x); },
      [](double y, double x, double parentAdjoint) {
        return parentAdjoint * x / (y * y + x * x);
      },
      [](double y, double x, double parentAdjoint) {
        return parentAdjoint * -y / (y * y + x * x);
      },
      [](const IntrusiveSharedPtr<Expression>& y,
         const IntrusiveSharedPtr<Expression>& x,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * x / (y * y + x * x);
      },
      [](const IntrusiveSharedPtr<Expression>& y,
         const IntrusiveSharedPtr<Expression>& x,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * -y / (y * y + x * x);
      },
      y, x);
}

IntrusiveSharedPtr<Expression> cos(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x) {
  if (x == Zero()) {
    return MakeConstant(1.0);
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::cos(x); },
      [](double x, double, double parentAdjoint) {
        return -parentAdjoint * std::sin(x);
      },
      [](const IntrusiveSharedPtr<Expression>& x,
         const IntrusiveSharedPtr<Expression>&,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * -sleipnir::sin(x);
      },
      x);
}

IntrusiveSharedPtr<Expression> cosh(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x) {
  if (x == Zero()) {
    return MakeConstant(1.0);
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::cosh(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint * std::sinh(x);
      },
      [](const IntrusiveSharedPtr<Expression>& x,
         const IntrusiveSharedPtr<Expression>&,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * sleipnir::sinh(x);
      },
      x);
}

IntrusiveSharedPtr<Expression> erf(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x) {
  static constexpr double sqrt_pi =
      1.7724538509055160272981674833411451872554456638435L;

  if (x == Zero()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::erf(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint * 2.0 / sqrt_pi * std::exp(-x * x);
      },
      [](const IntrusiveSharedPtr<Expression>& x,
         const IntrusiveSharedPtr<Expression>&,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * 2.0 / sqrt_pi * sleipnir::exp(-x * x);
      },
      x);
}

IntrusiveSharedPtr<Expression> exp(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x) {
  if (x == Zero()) {
    return MakeConstant(1.0);
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::exp(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint * std::exp(x);
      },
      [](const IntrusiveSharedPtr<Expression>& x,
         const IntrusiveSharedPtr<Expression>&,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * sleipnir::exp(x);
      },
      x);
}

IntrusiveSharedPtr<Expression> hypot(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x,
    const IntrusiveSharedPtr<Expression>& y) {
  if (x == Zero() && y == Zero()) {
    return Zero();
  }

  if (x == Zero() && y != Zero()) {
    // Evaluate the expression's type
    ExpressionType type;
    if (y->type == ExpressionType::kConstant) {
      type = ExpressionType::kConstant;
    } else {
      type = ExpressionType::kNonlinear;
    }

    return AllocateIntrusiveShared<Expression>(
        GlobalPoolAllocator<Expression>(), type,
        [](double x, double y) { return std::hypot(x, y); },
        [](double x, double y, double parentAdjoint) {
          return parentAdjoint * x / std::hypot(x, y);
        },
        [](double x, double y, double parentAdjoint) {
          return parentAdjoint * y / std::hypot(x, y);
        },
        [](const IntrusiveSharedPtr<Expression>& x,
           const IntrusiveSharedPtr<Expression>& y,
           const IntrusiveSharedPtr<Expression>& parentAdjoint) {
          return parentAdjoint * x / sleipnir::hypot(x, y);
        },
        [](const IntrusiveSharedPtr<Expression>& x,
           const IntrusiveSharedPtr<Expression>& y,
           const IntrusiveSharedPtr<Expression>& parentAdjoint) {
          return parentAdjoint * y / sleipnir::hypot(x, y);
        },
        MakeConstant(0.0), y);
  } else if (x != Zero() && y == Zero()) {
    // Evaluate the expression's type
    ExpressionType type;
    if (x->type == ExpressionType::kConstant) {
      type = ExpressionType::kConstant;
    } else {
      type = ExpressionType::kNonlinear;
    }

    return AllocateIntrusiveShared<Expression>(
        GlobalPoolAllocator<Expression>(), type,
        [](double x, double y) { return std::hypot(x, y); },
        [](double x, double y, double parentAdjoint) {
          return parentAdjoint * x / std::hypot(x, y);
        },
        [](double x, double y, double parentAdjoint) {
          return parentAdjoint * y / std::hypot(x, y);
        },
        [](const IntrusiveSharedPtr<Expression>& x,
           const IntrusiveSharedPtr<Expression>& y,
           const IntrusiveSharedPtr<Expression>& parentAdjoint) {
          return parentAdjoint * x / sleipnir::hypot(x, y);
        },
        [](const IntrusiveSharedPtr<Expression>& x,
           const IntrusiveSharedPtr<Expression>& y,
           const IntrusiveSharedPtr<Expression>& parentAdjoint) {
          return parentAdjoint * y / sleipnir::hypot(x, y);
        },
        x, MakeConstant(0.0));
  } else {
    // Evaluate the expression's type
    ExpressionType type;
    if (x->type == ExpressionType::kConstant &&
        y->type == ExpressionType::kConstant) {
      type = ExpressionType::kConstant;
    } else {
      type = ExpressionType::kNonlinear;
    }

    return AllocateIntrusiveShared<Expression>(
        GlobalPoolAllocator<Expression>(), type,
        [](double x, double y) { return std::hypot(x, y); },
        [](double x, double y, double parentAdjoint) {
          return parentAdjoint * x / std::hypot(x, y);
        },
        [](double x, double y, double parentAdjoint) {
          return parentAdjoint * y / std::hypot(x, y);
        },
        [](const IntrusiveSharedPtr<Expression>& x,
           const IntrusiveSharedPtr<Expression>& y,
           const IntrusiveSharedPtr<Expression>& parentAdjoint) {
          return parentAdjoint * x / sleipnir::hypot(x, y);
        },
        [](const IntrusiveSharedPtr<Expression>& x,
           const IntrusiveSharedPtr<Expression>& y,
           const IntrusiveSharedPtr<Expression>& parentAdjoint) {
          return parentAdjoint * y / sleipnir::hypot(x, y);
        },
        x, y);
  }
}

IntrusiveSharedPtr<Expression> log(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x) {
  if (x == Zero()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::log(x); },
      [](double x, double, double parentAdjoint) { return parentAdjoint / x; },
      [](const IntrusiveSharedPtr<Expression>& x,
         const IntrusiveSharedPtr<Expression>&,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint / x;
      },
      x);
}

IntrusiveSharedPtr<Expression> log10(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x) {
  static constexpr double ln10 = 2.3025850929940456840179914546843L;

  if (x == Zero()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::log10(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / (ln10 * x);
      },
      [](const IntrusiveSharedPtr<Expression>& x,
         const IntrusiveSharedPtr<Expression>&,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint / (ln10 * x);
      },
      x);
}

IntrusiveSharedPtr<Expression> pow(  // NOLINT
    const IntrusiveSharedPtr<Expression>& base,
    const IntrusiveSharedPtr<Expression>& power) {
  if (base == Zero()) {
    return Zero();
  }
  if (power == Zero()) {
    return MakeConstant(1.0);
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (base->type == ExpressionType::kConstant &&
      power->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else if (power->type == ExpressionType::kConstant && power->value == 0.0) {
    type = ExpressionType::kConstant;
  } else if (base->type == ExpressionType::kLinear &&
             power->type == ExpressionType::kConstant && power->value == 1.0) {
    type = ExpressionType::kLinear;
  } else if (base->type == ExpressionType::kLinear &&
             power->type == ExpressionType::kConstant && power->value == 2.0) {
    type = ExpressionType::kQuadratic;
  } else if (base->type == ExpressionType::kQuadratic &&
             power->type == ExpressionType::kConstant && power->value == 1.0) {
    type = ExpressionType::kQuadratic;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
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
      [](const IntrusiveSharedPtr<Expression>& base,
         const IntrusiveSharedPtr<Expression>& power,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * sleipnir::pow(base, power - 1) * power;
      },
      [](const IntrusiveSharedPtr<Expression>& base,
         const IntrusiveSharedPtr<Expression>& power,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        // Since x * std::log(x) -> 0 as x -> 0
        if (base->value == 0.0) {
          return Zero();
        } else {
          return parentAdjoint * sleipnir::pow(base, power - 1) * base *
                 sleipnir::log(base);
        }
      },
      base, power);
}

IntrusiveSharedPtr<Expression> sin(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x) {
  if (x == Zero()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::sin(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint * std::cos(x);
      },
      [](const IntrusiveSharedPtr<Expression>& x,
         const IntrusiveSharedPtr<Expression>&,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * sleipnir::cos(x);
      },
      x);
}

IntrusiveSharedPtr<Expression> sinh(const IntrusiveSharedPtr<Expression>& x) {
  if (x == Zero()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::sinh(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint * std::cosh(x);
      },
      [](const IntrusiveSharedPtr<Expression>& x,
         const IntrusiveSharedPtr<Expression>&,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint * sleipnir::cosh(x);
      },
      x);
}

IntrusiveSharedPtr<Expression> sqrt(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x) {
  if (x == Zero()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::sqrt(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / (2.0 * std::sqrt(x));
      },
      [](const IntrusiveSharedPtr<Expression>& x,
         const IntrusiveSharedPtr<Expression>&,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint / (2.0 * sleipnir::sqrt(x));
      },
      x);
}

IntrusiveSharedPtr<Expression> tan(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x) {
  if (x == Zero()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::tan(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / (std::cos(x) * std::cos(x));
      },
      [](const IntrusiveSharedPtr<Expression>& x,
         const IntrusiveSharedPtr<Expression>&,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint / (sleipnir::cos(x) * sleipnir::cos(x));
      },
      x);
}

IntrusiveSharedPtr<Expression> tanh(const IntrusiveSharedPtr<Expression>& x) {
  if (x == Zero()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x->type == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::tanh(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / (std::cosh(x) * std::cosh(x));
      },
      [](const IntrusiveSharedPtr<Expression>& x,
         const IntrusiveSharedPtr<Expression>&,
         const IntrusiveSharedPtr<Expression>& parentAdjoint) {
        return parentAdjoint / (sleipnir::cosh(x) * sleipnir::cosh(x));
      },
      x);
}

}  // namespace sleipnir
