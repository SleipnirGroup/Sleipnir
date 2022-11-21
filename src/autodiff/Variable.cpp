// Copyright (c) Joshua Nichols and Tyler Veness

#include "sleipnir/autodiff/Variable.hpp"

#include <cmath>
#include <tuple>
#include <vector>
#include <fmt/core.h>

#include "sleipnir/SymbolExports.hpp"

namespace sleipnir::autodiff {

Variable::Variable(double value)
    : expr{AllocateIntrusiveShared<Expression>(Allocator(), value)} {}

Variable::Variable(int value)
    : expr{AllocateIntrusiveShared<Expression>(Allocator(), value)} {}

Variable::Variable(IntrusiveSharedPtr<Expression> expr)
    : expr{std::move(expr)} {}

Variable& Variable::operator=(double value) {
  if (expr == nullptr) {
    expr = AllocateIntrusiveShared<Expression>(Allocator(), value);
  } else {
    if (expr->args[0] != nullptr) {
      fmt::print(stderr, 
                 "WARNING {}:{}: Modified the value of a dependent variable", 
                 __FILE__, __LINE__);
    }
    expr->value = value;
  }
  return *this;
}

Variable& Variable::operator=(int value) {
  if (expr == nullptr) {
    expr = AllocateIntrusiveShared<Expression>(Allocator(), value);
  } else{
    if (expr->args[0] != nullptr) {
      fmt::print(stderr, 
                 "WARNING {}:{}: Modified the value of a dependent variable",
                 __FILE__, __LINE__);
    }
    expr->value = value;
  }
  return *this;
}

SLEIPNIR_DLLEXPORT Variable operator*(double lhs, const Variable& rhs) {
  return Variable{lhs * rhs.expr};
}

SLEIPNIR_DLLEXPORT Variable operator*(const Variable& lhs, double rhs) {
  return Variable{lhs.expr * rhs};
}

SLEIPNIR_DLLEXPORT Variable operator*(const Variable& lhs,
                                      const Variable& rhs) {
  return Variable{lhs.expr * rhs.expr};
}

Variable& Variable::operator*=(double rhs) {
  *this = *this * rhs;
  return *this;
}

Variable& Variable::operator*=(const Variable& rhs) {
  *this = *this * rhs;
  return *this;
}

SLEIPNIR_DLLEXPORT Variable operator/(double lhs, const Variable& rhs) {
  return Variable{lhs / rhs.expr};
}

SLEIPNIR_DLLEXPORT Variable operator/(const Variable& lhs, double rhs) {
  return Variable{lhs.expr / rhs};
}

SLEIPNIR_DLLEXPORT Variable operator/(const Variable& lhs,
                                      const Variable& rhs) {
  return Variable{lhs.expr / rhs.expr};
}

Variable& Variable::operator/=(double rhs) {
  *this = *this / rhs;
  return *this;
}

Variable& Variable::operator/=(const Variable& rhs) {
  *this = *this / rhs;
  return *this;
}

SLEIPNIR_DLLEXPORT Variable operator+(double lhs, const Variable& rhs) {
  return Variable{lhs + rhs.expr};
}

SLEIPNIR_DLLEXPORT Variable operator+(const Variable& lhs, double rhs) {
  return Variable{lhs.expr + rhs};
}

SLEIPNIR_DLLEXPORT Variable operator+(const Variable& lhs,
                                      const Variable& rhs) {
  return Variable{lhs.expr + rhs.expr};
}

Variable& Variable::operator+=(double rhs) {
  *this = *this + rhs;
  return *this;
}

Variable& Variable::operator+=(const Variable& rhs) {
  *this = *this + rhs;
  return *this;
}

SLEIPNIR_DLLEXPORT Variable operator-(double lhs, const Variable& rhs) {
  return Variable{lhs - rhs.expr};
}

SLEIPNIR_DLLEXPORT Variable operator-(const Variable& lhs, double rhs) {
  return Variable{lhs.expr - rhs};
}

SLEIPNIR_DLLEXPORT Variable operator-(const Variable& lhs,
                                      const Variable& rhs) {
  return Variable{lhs.expr - rhs.expr};
}

Variable& Variable::operator-=(double rhs) {
  *this = *this - rhs;
  return *this;
}

Variable& Variable::operator-=(const Variable& rhs) {
  *this = *this - rhs;
  return *this;
}

SLEIPNIR_DLLEXPORT Variable operator-(const Variable& lhs) {
  return Variable{-lhs.expr};
}

SLEIPNIR_DLLEXPORT Variable operator+(const Variable& lhs) {
  return Variable{+lhs.expr};
}

SLEIPNIR_DLLEXPORT bool operator==(double lhs, const Variable& rhs) {
  return std::abs(lhs - rhs.Value()) < 1e-9;
}

SLEIPNIR_DLLEXPORT bool operator==(const Variable& lhs, double rhs) {
  return std::abs(lhs.Value() - rhs) < 1e-9;
}

SLEIPNIR_DLLEXPORT bool operator==(const Variable& lhs, const Variable& rhs) {
  return std::abs(lhs.Value() - rhs.Value()) < 1e-9;
}

SLEIPNIR_DLLEXPORT bool operator!=(double lhs, const Variable& rhs) {
  return lhs != rhs.Value();
}

SLEIPNIR_DLLEXPORT bool operator!=(const Variable& lhs, double rhs) {
  return lhs.Value() != rhs;
}

SLEIPNIR_DLLEXPORT bool operator!=(const Variable& lhs, const Variable& rhs) {
  return lhs.Value() != rhs.Value();
}

SLEIPNIR_DLLEXPORT bool operator<(double lhs, const Variable& rhs) {
  return lhs < rhs.Value();
}

SLEIPNIR_DLLEXPORT bool operator<(const Variable& lhs, double rhs) {
  return lhs.Value() < rhs;
}

SLEIPNIR_DLLEXPORT bool operator<(const Variable& lhs, const Variable& rhs) {
  return lhs.Value() < rhs.Value();
}

SLEIPNIR_DLLEXPORT bool operator>(double lhs, const Variable& rhs) {
  return lhs > rhs.Value();
}

SLEIPNIR_DLLEXPORT bool operator>(const Variable& lhs, double rhs) {
  return lhs.Value() > rhs;
}

SLEIPNIR_DLLEXPORT bool operator>(const Variable& lhs, const Variable& rhs) {
  return lhs.Value() > rhs.Value();
}

SLEIPNIR_DLLEXPORT bool operator<=(double lhs, const Variable& rhs) {
  return lhs <= rhs.Value();
}

SLEIPNIR_DLLEXPORT bool operator<=(const Variable& lhs, double rhs) {
  return lhs.Value() <= rhs;
}

SLEIPNIR_DLLEXPORT bool operator<=(const Variable& lhs, const Variable& rhs) {
  return lhs.Value() <= rhs.Value();
}

SLEIPNIR_DLLEXPORT bool operator>=(double lhs, const Variable& rhs) {
  return lhs >= rhs.Value();
}

SLEIPNIR_DLLEXPORT bool operator>=(const Variable& lhs, double rhs) {
  return lhs.Value() >= rhs;
}

SLEIPNIR_DLLEXPORT bool operator>=(const Variable& lhs, const Variable& rhs) {
  return lhs.Value() >= rhs.Value();
}

double Variable::Value() const {
  if (expr == nullptr) {
    return 0.0;
  } else {
    return expr->value;
  }
}

ExpressionType Variable::Type() const {
  if (expr == nullptr) {
    return ExpressionType::kNone;
  } else {
    return expr->type;
  }
}

void Variable::Update() {
  if (expr != nullptr) {
    expr->Update();
  }
}

Variable abs(double x) {
  return Variable{autodiff::abs(autodiff::MakeConstant(x))};
}

Variable abs(const Variable& x) {
  return Variable{abs(x.expr)};
}

Variable acos(double x) {
  return Variable{autodiff::acos(autodiff::MakeConstant(x))};
}

Variable acos(const Variable& x) {
  return Variable{acos(x.expr)};
}

Variable asin(double x) {
  return Variable{autodiff::asin(autodiff::MakeConstant(x))};
}

Variable asin(const Variable& x) {
  return Variable{asin(x.expr)};
}

Variable atan(double x) {
  return Variable{autodiff::atan(MakeConstant(x))};
}

Variable atan(const Variable& x) {
  return Variable{atan(x.expr)};
}

Variable atan2(double y, const Variable& x) {
  return Variable{autodiff::atan2(MakeConstant(y), x.expr)};
}

Variable atan2(const Variable& y, double x) {
  return Variable{autodiff::atan2(y.expr, MakeConstant(x))};
}

Variable atan2(const Variable& y, const Variable& x) {
  return Variable{atan2(y.expr, x.expr)};
}

Variable cos(double x) {
  return Variable{autodiff::cos(MakeConstant(x))};
}

Variable cos(const Variable& x) {
  return Variable{cos(x.expr)};
}

Variable cosh(double x) {
  return Variable{autodiff::cosh(MakeConstant(x))};
}

Variable cosh(const Variable& x) {
  return Variable{cosh(x.expr)};
}

Variable erf(double x) {
  return Variable{autodiff::erf(MakeConstant(x))};
}

Variable erf(const Variable& x) {
  return Variable{erf(x.expr)};
}

Variable exp(double x) {
  return Variable{autodiff::exp(MakeConstant(x))};
}

Variable exp(const Variable& x) {
  return Variable{exp(x.expr)};
}

Variable hypot(double x, const Variable& y) {
  return Variable{autodiff::hypot(MakeConstant(x), y.expr)};
}

Variable hypot(const Variable& x, double y) {
  return Variable{autodiff::hypot(x.expr, MakeConstant(y))};
}

Variable hypot(const Variable& x, const Variable& y) {
  return Variable{hypot(x.expr, y.expr)};
}

Variable log(double x) {
  return Variable{autodiff::log(MakeConstant(x))};
}

Variable log(const Variable& x) {
  return Variable{log(x.expr)};
}

Variable log10(double x) {
  return Variable{autodiff::log10(MakeConstant(x))};
}

Variable log10(const Variable& x) {
  return Variable{log10(x.expr)};
}

Variable pow(double base, const Variable& power) {
  return Variable{autodiff::pow(MakeConstant(base), power.expr)};
}

Variable pow(const Variable& base, double power) {
  return Variable{autodiff::pow(base.expr, MakeConstant(power))};
}

Variable pow(const Variable& base, const Variable& power) {
  return Variable{pow(base.expr, power.expr)};
}

Variable sin(double x) {
  return Variable{autodiff::sin(MakeConstant(x))};
}

Variable sin(const Variable& x) {
  return Variable{sin(x.expr)};
}

Variable sinh(double x) {
  return Variable{autodiff::sinh(MakeConstant(x))};
}

Variable sinh(const Variable& x) {
  return Variable{sinh(x.expr)};
}

Variable sqrt(double x) {
  return Variable{autodiff::sqrt(MakeConstant(x))};
}

Variable sqrt(const Variable& x) {
  return Variable{sqrt(x.expr)};
}

Variable tan(double x) {
  return Variable{autodiff::tan(MakeConstant(x))};
}

Variable tan(const Variable& x) {
  return Variable{tan(x.expr)};
}

Variable tanh(double x) {
  return Variable{autodiff::tanh(MakeConstant(x))};
}

Variable tanh(const Variable& x) {
  return Variable{tanh(x.expr)};
}

}  // namespace sleipnir::autodiff
