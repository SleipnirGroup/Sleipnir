// Copyright (c) Joshua Nichols and Tyler Veness

#include "sleipnir/autodiff/Variable.hpp"

#include <cmath>
#include <tuple>
#include <vector>

#include <fmt/core.h>

#include "sleipnir/SymbolExports.hpp"
#include "sleipnir/autodiff/ExpressionGraph.hpp"

namespace sleipnir {

Variable::Variable(double value)
    : expr{AllocateIntrusiveShared<Expression>(
          GlobalPoolAllocator<Expression>(), value)} {}

Variable::Variable(int value)
    : expr{AllocateIntrusiveShared<Expression>(
          GlobalPoolAllocator<Expression>(), value)} {}

Variable::Variable(IntrusiveSharedPtr<Expression> expr)
    : expr{std::move(expr)} {}

Variable& Variable::operator=(double value) {
  if (expr == Zero()) {
    expr = AllocateIntrusiveShared<Expression>(
        GlobalPoolAllocator<Expression>(), value);
  } else {
    if (expr->args[0] != Zero()) {
      fmt::print(stderr,
                 "WARNING: {}:{}: Modified the value of a dependent variable\n",
                 __FILE__, __LINE__);
    }
    expr->value = value;
  }
  return *this;
}

Variable& Variable::operator=(int value) {
  if (expr == Zero()) {
    expr = AllocateIntrusiveShared<Expression>(
        GlobalPoolAllocator<Expression>(), value);
  } else {
    if (expr->args[0] != Zero()) {
      fmt::print(stderr,
                 "WARNING: {}:{}: Modified the value of a dependent variable\n",
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
  return expr->value;
}

ExpressionType Variable::Type() const {
  return expr->type;
}

void Variable::Update() {
  if (expr != Zero()) {
    ExpressionGraph graph{*this};
    graph.Update();
  }
}

Variable abs(double x) {
  return Variable{sleipnir::abs(MakeConstant(x))};
}

Variable abs(const Variable& x) {
  return Variable{abs(x.expr)};
}

Variable acos(double x) {
  return Variable{sleipnir::cos(MakeConstant(x))};
}

Variable acos(const Variable& x) {
  return Variable{acos(x.expr)};
}

Variable asin(double x) {
  return Variable{sleipnir::asin(MakeConstant(x))};
}

Variable asin(const Variable& x) {
  return Variable{asin(x.expr)};
}

Variable atan(double x) {
  return Variable{sleipnir::atan(MakeConstant(x))};
}

Variable atan(const Variable& x) {
  return Variable{atan(x.expr)};
}

Variable atan2(double y, const Variable& x) {
  return Variable{sleipnir::atan2(MakeConstant(y), x.expr)};
}

Variable atan2(const Variable& y, double x) {
  return Variable{sleipnir::atan2(y.expr, MakeConstant(x))};
}

Variable atan2(const Variable& y, const Variable& x) {
  return Variable{atan2(y.expr, x.expr)};
}

Variable cos(double x) {
  return Variable{sleipnir::cos(MakeConstant(x))};
}

Variable cos(const Variable& x) {
  return Variable{cos(x.expr)};
}

Variable cosh(double x) {
  return Variable{sleipnir::cosh(MakeConstant(x))};
}

Variable cosh(const Variable& x) {
  return Variable{cosh(x.expr)};
}

Variable erf(double x) {
  return Variable{sleipnir::erf(MakeConstant(x))};
}

Variable erf(const Variable& x) {
  return Variable{erf(x.expr)};
}

Variable exp(double x) {
  return Variable{sleipnir::exp(MakeConstant(x))};
}

Variable exp(const Variable& x) {
  return Variable{exp(x.expr)};
}

Variable hypot(double x, const Variable& y) {
  return Variable{sleipnir::hypot(MakeConstant(x), y.expr)};
}

Variable hypot(const Variable& x, double y) {
  return Variable{sleipnir::hypot(x.expr, MakeConstant(y))};
}

Variable hypot(const Variable& x, const Variable& y) {
  return Variable{sleipnir::hypot(x.expr, y.expr)};
}

Variable log(double x) {
  return Variable{sleipnir::log(MakeConstant(x))};
}

Variable log(const Variable& x) {
  return Variable{log(x.expr)};
}

Variable log10(double x) {
  return Variable{sleipnir::log10(MakeConstant(x))};
}

Variable log10(const Variable& x) {
  return Variable{log10(x.expr)};
}

Variable pow(double base, const Variable& power) {
  return Variable{sleipnir::pow(MakeConstant(base), power.expr)};
}

Variable pow(const Variable& base, double power) {
  return Variable{sleipnir::pow(base.expr, MakeConstant(power))};
}

Variable pow(const Variable& base, const Variable& power) {
  return Variable{pow(base.expr, power.expr)};
}

Variable sin(double x) {
  return Variable{sleipnir::sin(MakeConstant(x))};
}

Variable sin(const Variable& x) {
  return Variable{sin(x.expr)};
}

Variable sinh(double x) {
  return Variable{sleipnir::sinh(MakeConstant(x))};
}

Variable sinh(const Variable& x) {
  return Variable{sinh(x.expr)};
}

Variable sqrt(double x) {
  return Variable{sleipnir::sqrt(MakeConstant(x))};
}

Variable sqrt(const Variable& x) {
  return Variable{sqrt(x.expr)};
}

Variable tan(double x) {
  return Variable{sleipnir::tan(MakeConstant(x))};
}

Variable tan(const Variable& x) {
  return Variable{tan(x.expr)};
}

Variable tanh(double x) {
  return Variable{sleipnir::tanh(MakeConstant(x))};
}

Variable tanh(const Variable& x) {
  return Variable{tanh(x.expr)};
}

}  // namespace sleipnir
