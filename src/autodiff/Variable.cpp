// Copyright (c) Sleipnir contributors

#include "sleipnir/autodiff/Variable.hpp"

#include <fmt/core.h>

#include "sleipnir/autodiff/Expression.hpp"
#include "sleipnir/autodiff/ExpressionGraph.hpp"

namespace sleipnir {

Variable::Variable(double value) : expr{detail::MakeExpressionPtr(value)} {}

Variable::Variable(int value) : expr{detail::MakeExpressionPtr(value)} {}

Variable::Variable(const detail::ExpressionPtr& expr) : expr{expr} {}

Variable::Variable(detail::ExpressionPtr&& expr) : expr{std::move(expr)} {}

Variable& Variable::operator=(double value) {
  expr = detail::MakeExpressionPtr(value);

  return *this;
}

Variable& Variable::operator=(int value) {
  expr = detail::MakeExpressionPtr(value);

  return *this;
}

Variable& Variable::SetValue(double value) {
  if (expr == detail::Zero()) {
    expr = detail::MakeExpressionPtr(value);
  } else {
    // We only need to check the first argument since unary and binary operators
    // both use it
    if (expr->args[0] != detail::Zero()) {
      fmt::print(stderr,
                 "WARNING: {}:{}: Modified the value of a dependent variable\n",
                 __FILE__, __LINE__);
    }
    expr->value = value;
  }
  return *this;
}

Variable& Variable::SetValue(int value) {
  if (expr == detail::Zero()) {
    expr = detail::MakeExpressionPtr(value);
  } else {
    // We only need to check the first argument since unary and binary operators
    // both use it
    if (expr->args[0] != detail::Zero()) {
      fmt::print(stderr,
                 "WARNING: {}:{}: Modified the value of a dependent variable\n",
                 __FILE__, __LINE__);
    }
    expr->value = value;
  }
  return *this;
}

Variable operator*(const Variable& lhs, const Variable& rhs) {
  return Variable{lhs.expr * rhs.expr};
}

Variable& Variable::operator*=(const Variable& rhs) {
  *this = *this * rhs;
  return *this;
}

Variable operator/(const Variable& lhs, const Variable& rhs) {
  return Variable{lhs.expr / rhs.expr};
}

Variable& Variable::operator/=(const Variable& rhs) {
  *this = *this / rhs;
  return *this;
}

Variable operator+(const Variable& lhs, const Variable& rhs) {
  return Variable{lhs.expr + rhs.expr};
}

Variable& Variable::operator+=(const Variable& rhs) {
  *this = *this + rhs;
  return *this;
}

Variable operator-(const Variable& lhs, const Variable& rhs) {
  return Variable{lhs.expr - rhs.expr};
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
  if (expr != detail::Zero()) {
    detail::ExpressionGraph graph{*this};
    graph.Update();
  }
}

Variable abs(const Variable& x) {
  return Variable{detail::abs(x.expr)};
}

Variable acos(const Variable& x) {
  return Variable{detail::acos(x.expr)};
}

Variable asin(const Variable& x) {
  return Variable{detail::asin(x.expr)};
}

Variable atan(const Variable& x) {
  return Variable{detail::atan(x.expr)};
}

Variable atan2(const Variable& y, const Variable& x) {
  return Variable{detail::atan2(y.expr, x.expr)};
}

Variable cos(const Variable& x) {
  return Variable{detail::cos(x.expr)};
}

Variable cosh(const Variable& x) {
  return Variable{detail::cosh(x.expr)};
}

Variable erf(const Variable& x) {
  return Variable{detail::erf(x.expr)};
}

Variable exp(const Variable& x) {
  return Variable{detail::exp(x.expr)};
}

Variable hypot(const Variable& x, const Variable& y) {
  return Variable{detail::hypot(x.expr, y.expr)};
}

Variable hypot(const Variable& x, const Variable& y, const Variable& z) {
  return Variable{sleipnir::sqrt(sleipnir::pow(x, 2) + sleipnir::pow(y, 2) +
                                 sleipnir::pow(z, 2))};
}

Variable log(const Variable& x) {
  return Variable{detail::log(x.expr)};
}

Variable log10(const Variable& x) {
  return Variable{detail::log10(x.expr)};
}

Variable pow(const Variable& base, const Variable& power) {
  return Variable{detail::pow(base.expr, power.expr)};
}

Variable sign(const Variable& x) {
  return Variable{detail::sign(x.expr)};
}

Variable sin(const Variable& x) {
  return Variable{detail::sin(x.expr)};
}

Variable sinh(const Variable& x) {
  return Variable{detail::sinh(x.expr)};
}

Variable sqrt(const Variable& x) {
  return Variable{detail::sqrt(x.expr)};
}

Variable tan(const Variable& x) {
  return Variable{detail::tan(x.expr)};
}

Variable tanh(const Variable& x) {
  return Variable{detail::tanh(x.expr)};
}

}  // namespace sleipnir
