// Copyright (c) Sleipnir contributors

#include "sleipnir/autodiff/Variable.hpp"

#include <fmt/core.h>

#include "sleipnir/SymbolExports.hpp"
#include "sleipnir/autodiff/Expression.hpp"
#include "sleipnir/autodiff/ExpressionGraph.hpp"

namespace sleipnir {

Variable::Variable(double value)
    : expr{MakeExpression(value, ExpressionType::kConstant)} {}

Variable::Variable(int value)
    : expr{MakeExpression(value, ExpressionType::kConstant)} {}

Variable::Variable(IntrusiveSharedPtr<Expression> expr)
    : expr{std::move(expr)} {}

Variable& Variable::operator=(double value) {
  expr = MakeExpression(value, ExpressionType::kConstant);

  return *this;
}

Variable& Variable::operator=(int value) {
  expr = MakeExpression(value, ExpressionType::kConstant);

  return *this;
}

Variable& Variable::SetValue(double value) {
  if (expr == Zero()) {
    expr = MakeExpression(value);
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

Variable& Variable::SetValue(int value) {
  if (expr == Zero()) {
    expr = MakeExpression(value);
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
  return Variable{MakeExpression(x, ExpressionType::kConstant)};
}

Variable abs(double x) {
  return sleipnir::abs(Constant(x));
}

Variable abs(const Variable& x) {
  return Variable{abs(x.expr)};
}

Variable acos(double x) {
  return sleipnir::acos(Constant(x));
}

Variable acos(const Variable& x) {
  return Variable{acos(x.expr)};
}

Variable asin(double x) {
  return sleipnir::asin(Constant(x));
}

Variable asin(const Variable& x) {
  return Variable{asin(x.expr)};
}

Variable atan(double x) {
  return sleipnir::atan(Constant(x));
}

Variable atan(const Variable& x) {
  return Variable{atan(x.expr)};
}

Variable atan2(double y, const Variable& x) {
  return sleipnir::atan2(Constant(y), x);
}

Variable atan2(const Variable& y, double x) {
  return sleipnir::atan2(y, Constant(x));
}

Variable atan2(const Variable& y, const Variable& x) {
  return Variable{atan2(y.expr, x.expr)};
}

Variable cos(double x) {
  return sleipnir::cos(Constant(x));
}

Variable cos(const Variable& x) {
  return Variable{cos(x.expr)};
}

Variable cosh(double x) {
  return sleipnir::cosh(Constant(x));
}

Variable cosh(const Variable& x) {
  return Variable{cosh(x.expr)};
}

Variable erf(double x) {
  return sleipnir::erf(Constant(x));
}

Variable erf(const Variable& x) {
  return Variable{erf(x.expr)};
}

Variable exp(double x) {
  return sleipnir::exp(Constant(x));
}

Variable exp(const Variable& x) {
  return Variable{exp(x.expr)};
}

Variable hypot(double x, const Variable& y) {
  return sleipnir::hypot(Constant(x), y);
}

Variable hypot(const Variable& x, double y) {
  return sleipnir::hypot(x, Constant(y));
}

Variable hypot(const Variable& x, const Variable& y) {
  return Variable{sleipnir::hypot(x.expr, y.expr)};
}

Variable hypot(const Variable& x, const Variable& y, const Variable& z) {
  return Variable{sleipnir::sqrt(sleipnir::pow(x, 2) + sleipnir::pow(y, 2) +
                                 sleipnir::pow(z, 2))};
}

Variable log(double x) {
  return sleipnir::log(Constant(x));
}

Variable log(const Variable& x) {
  return Variable{log(x.expr)};
}

Variable log10(double x) {
  return sleipnir::log10(Constant(x));
}

Variable log10(const Variable& x) {
  return Variable{log10(x.expr)};
}

Variable pow(double base, const Variable& power) {
  return sleipnir::pow(Constant(base), power);
}

Variable pow(const Variable& base, double power) {
  return sleipnir::pow(base, Constant(power));
}

Variable pow(const Variable& base, const Variable& power) {
  return Variable{pow(base.expr, power.expr)};
}

Variable sign(double x) {
  return sleipnir::sign(Constant(x));
}

Variable sign(const Variable& x) {
  return Variable{sign(x.expr)};
}

Variable sin(double x) {
  return sleipnir::sin(Constant(x));
}

Variable sin(const Variable& x) {
  return Variable{sin(x.expr)};
}

Variable sinh(double x) {
  return sleipnir::sinh(Constant(x));
}

Variable sinh(const Variable& x) {
  return Variable{sinh(x.expr)};
}

Variable sqrt(double x) {
  return sleipnir::sqrt(Constant(x));
}

Variable sqrt(const Variable& x) {
  return Variable{sqrt(x.expr)};
}

Variable tan(double x) {
  return sleipnir::tan(Constant(x));
}

Variable tan(const Variable& x) {
  return Variable{tan(x.expr)};
}

Variable tanh(double x) {
  return sleipnir::tanh(Constant(x));
}

Variable tanh(const Variable& x) {
  return Variable{tanh(x.expr)};
}

}  // namespace sleipnir
