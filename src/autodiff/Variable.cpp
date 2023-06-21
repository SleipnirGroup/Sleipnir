// Copyright (c) Sleipnir contributors

#include "sleipnir/autodiff/Variable.hpp"

#include <fmt/core.h>

#include "sleipnir/SymbolExports.hpp"
#include "sleipnir/autodiff/Expression.hpp"
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

Variable operator*(double lhs, const Variable& rhs) {
  return Variable{lhs * rhs.expr};
}

Variable operator*(const Variable& lhs, double rhs) {
  return Variable{lhs.expr * rhs};
}

Variable operator*(const Variable& lhs, const Variable& rhs) {
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

Variable operator/(double lhs, const Variable& rhs) {
  return Variable{lhs / rhs.expr};
}

Variable operator/(const Variable& lhs, double rhs) {
  return Variable{lhs.expr / rhs};
}

Variable operator/(const Variable& lhs, const Variable& rhs) {
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

Variable operator+(double lhs, const Variable& rhs) {
  return Variable{lhs + rhs.expr};
}

Variable operator+(const Variable& lhs, double rhs) {
  return Variable{lhs.expr + rhs};
}

Variable operator+(const Variable& lhs, const Variable& rhs) {
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

Variable operator-(double lhs, const Variable& rhs) {
  return Variable{lhs - rhs.expr};
}

Variable operator-(const Variable& lhs, double rhs) {
  return Variable{lhs.expr - rhs};
}

Variable operator-(const Variable& lhs, const Variable& rhs) {
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

Variable operator-(const Variable& lhs) {
  return Variable{-lhs.expr};
}

Variable operator+(const Variable& lhs) {
  return Variable{+lhs.expr};
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

Variable Constant(double x) {
  return Variable{ConstantExpr(x)};
}

Variable abs(double x) {
  return Variable{sleipnir::abs(ConstantExpr(x))};
}

Variable abs(const Variable& x) {
  return Variable{abs(x.expr)};
}

Variable acos(double x) {
  return Variable{sleipnir::acos(ConstantExpr(x))};
}

Variable acos(const Variable& x) {
  return Variable{acos(x.expr)};
}

Variable asin(double x) {
  return Variable{sleipnir::asin(ConstantExpr(x))};
}

Variable asin(const Variable& x) {
  return Variable{asin(x.expr)};
}

Variable atan(double x) {
  return Variable{sleipnir::atan(ConstantExpr(x))};
}

Variable atan(const Variable& x) {
  return Variable{atan(x.expr)};
}

Variable atan2(double y, const Variable& x) {
  return Variable{sleipnir::atan2(ConstantExpr(y), x.expr)};
}

Variable atan2(const Variable& y, double x) {
  return Variable{sleipnir::atan2(y.expr, ConstantExpr(x))};
}

Variable atan2(const Variable& y, const Variable& x) {
  return Variable{atan2(y.expr, x.expr)};
}

Variable cos(double x) {
  return Variable{sleipnir::cos(ConstantExpr(x))};
}

Variable cos(const Variable& x) {
  return Variable{cos(x.expr)};
}

Variable cosh(double x) {
  return Variable{sleipnir::cosh(ConstantExpr(x))};
}

Variable cosh(const Variable& x) {
  return Variable{cosh(x.expr)};
}

Variable erf(double x) {
  return Variable{sleipnir::erf(ConstantExpr(x))};
}

Variable erf(const Variable& x) {
  return Variable{erf(x.expr)};
}

Variable exp(double x) {
  return Variable{sleipnir::exp(ConstantExpr(x))};
}

Variable exp(const Variable& x) {
  return Variable{exp(x.expr)};
}

Variable hypot(double x, const Variable& y) {
  return Variable{sleipnir::hypot(ConstantExpr(x), y.expr)};
}

Variable hypot(const Variable& x, double y) {
  return Variable{sleipnir::hypot(x.expr, ConstantExpr(y))};
}

Variable hypot(const Variable& x, const Variable& y) {
  return Variable{sleipnir::hypot(x.expr, y.expr)};
}

Variable log(double x) {
  return Variable{sleipnir::log(ConstantExpr(x))};
}

Variable log(const Variable& x) {
  return Variable{log(x.expr)};
}

Variable log10(double x) {
  return Variable{sleipnir::log10(ConstantExpr(x))};
}

Variable log10(const Variable& x) {
  return Variable{log10(x.expr)};
}

Variable pow(double base, const Variable& power) {
  return Variable{sleipnir::pow(ConstantExpr(base), power.expr)};
}

Variable pow(const Variable& base, double power) {
  return Variable{sleipnir::pow(base.expr, ConstantExpr(power))};
}

Variable pow(const Variable& base, const Variable& power) {
  return Variable{pow(base.expr, power.expr)};
}

Variable sign(double x) {
  return Variable{sleipnir::sign(ConstantExpr(x))};
}

Variable sign(const Variable& x) {
  return Variable{sign(x.expr)};
}

Variable sin(double x) {
  return Variable{sleipnir::sin(ConstantExpr(x))};
}

Variable sin(const Variable& x) {
  return Variable{sin(x.expr)};
}

Variable sinh(double x) {
  return Variable{sleipnir::sinh(ConstantExpr(x))};
}

Variable sinh(const Variable& x) {
  return Variable{sinh(x.expr)};
}

Variable sqrt(double x) {
  return Variable{sleipnir::sqrt(ConstantExpr(x))};
}

Variable sqrt(const Variable& x) {
  return Variable{sqrt(x.expr)};
}

Variable tan(double x) {
  return Variable{sleipnir::tan(ConstantExpr(x))};
}

Variable tan(const Variable& x) {
  return Variable{tan(x.expr)};
}

Variable tanh(double x) {
  return Variable{sleipnir::tanh(ConstantExpr(x))};
}

Variable tanh(const Variable& x) {
  return Variable{tanh(x.expr)};
}

}  // namespace sleipnir
