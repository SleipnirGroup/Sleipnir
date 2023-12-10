// Copyright (c) Sleipnir contributors

#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "sleipnir/IntrusiveSharedPtr.hpp"
#include "sleipnir/SymbolExports.hpp"
#include "sleipnir/autodiff/Expression.hpp"

namespace sleipnir {

/**
 * An autodiff variable pointing to an expression node.
 */
class SLEIPNIR_DLLEXPORT Variable {
 public:
  /// The expression node.
  IntrusiveSharedPtr<Expression> expr = Zero();

  /**
   * Constructs an uninitialized Variable.
   */
  constexpr Variable() = default;

  /**
   * Copy constructor.
   */
  Variable(const Variable&) = default;

  /**
   * Copy assignment operator.
   */
  Variable& operator=(const Variable&) = default;

  /**
   * Move constructor.
   */
  Variable(Variable&&) = default;

  /**
   * Move assignment operator.
   */
  Variable& operator=(Variable&&) = default;

  /**
   * Constructs a Variable from a double.
   *
   * @param value The value of the Variable.
   */
  Variable(double value);  // NOLINT

  /**
   * Constructs a Variable from an int.
   *
   * @param value The value of the Variable.
   */
  Variable(int value);  // NOLINT

  /**
   * Constructs a Variable pointing to the specified expression.
   *
   * @param expr The autodiff variable.
   */
  explicit Variable(IntrusiveSharedPtr<Expression> expr);

  /**
   * Assignment operator for double.
   *
   * @param value The value of the Variable.
   */
  Variable& operator=(double value);

  /**
   * Assignment operator for int.
   *
   * @param value The value of the Variable.
   */
  Variable& operator=(int value);

  /**
   * Sets Variable's internal value.
   *
   * @param value The value of the Variable.
   */
  Variable& SetValue(double value);

  /**
   * Sets Variable's internal value.
   *
   * @param value The value of the Variable.
   */
  Variable& SetValue(int value);

  /**
   * double-Variable multiplication operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT Variable operator*(double lhs, const Variable& rhs);

  /**
   * Variable-double multiplication operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT Variable operator*(const Variable& lhs, double rhs);

  /**
   * Variable-Variable multiplication operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT Variable operator*(const Variable& lhs,
                                               const Variable& rhs);

  /**
   * Variable-double compound multiplication operator.
   *
   * @param rhs Operator right-hand side.
   */
  Variable& operator*=(double rhs);

  /**
   * Variable-Variable compound multiplication operator.
   *
   * @param rhs Operator right-hand side.
   */
  Variable& operator*=(const Variable& rhs);

  /**
   * double-Variable division operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT Variable operator/(double lhs, const Variable& rhs);

  /**
   * Variable-double division operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT Variable operator/(const Variable& lhs, double rhs);

  /**
   * Variable-Variable division operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT Variable operator/(const Variable& lhs,
                                               const Variable& rhs);

  /**
   * Variable-double compound division operator.
   *
   * @param rhs Operator right-hand side.
   */
  Variable& operator/=(double rhs);

  /**
   * Variable-Variable compound division operator.
   *
   * @param rhs Operator right-hand side.
   */
  Variable& operator/=(const Variable& rhs);

  /**
   * double-Variable addition operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT Variable operator+(double lhs, const Variable& rhs);

  /**
   * Variable-double addition operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT Variable operator+(const Variable& lhs, double rhs);

  /**
   * Variable-Variable addition operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT Variable operator+(const Variable& lhs,
                                               const Variable& rhs);

  /**
   * Variable-double compound addition operator.
   *
   * @param rhs Operator right-hand side.
   */
  Variable& operator+=(double rhs);

  /**
   * Variable-Variable compound addition operator.
   *
   * @param rhs Operator right-hand side.
   */
  Variable& operator+=(const Variable& rhs);

  /**
   * double-Variable subtraction operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT Variable operator-(double lhs, const Variable& rhs);

  /**
   * Variable-double subtraction operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT Variable operator-(const Variable& lhs, double rhs);

  /**
   * Variable-Variable subtraction operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT Variable operator-(const Variable& lhs,
                                               const Variable& rhs);

  /**
   * Variable-double compound subtraction operator.
   *
   * @param rhs Operator right-hand side.
   */
  Variable& operator-=(double rhs);

  /**
   * Variable-Variable compound subtraction operator.
   *
   * @param rhs Operator right-hand side.
   */
  Variable& operator-=(const Variable& rhs);

  /**
   * Unary minus operator.
   *
   * @param lhs Operand for unary minus.
   */
  friend SLEIPNIR_DLLEXPORT Variable operator-(const Variable& lhs);

  /**
   * Unary plus operator.
   *
   * @param lhs Operand for unary plus.
   */
  friend SLEIPNIR_DLLEXPORT Variable operator+(const Variable& lhs);

  /**
   * Returns the value of this variable.
   */
  double Value() const;

  /**
   * Returns the type of this expression (constant, linear, quadratic, or
   * nonlinear).
   */
  ExpressionType Type() const;

  /**
   * Updates the value of this variable based on the values of its dependent
   * variables.
   */
  void Update();
};

using VectorXvar = Eigen::Vector<Variable, Eigen::Dynamic>;
using MapVectorXvar = Eigen::Map<VectorXvar>;

/**
 * std::abs() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable abs(double x);

/**
 * std::abs() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable abs(const Variable& x);

/**
 * std::acos() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable acos(double x);

/**
 * std::acos() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable acos(const Variable& x);

/**
 * std::asin() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable asin(double x);

/**
 * std::asin() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable asin(const Variable& x);

/**
 * std::atan() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable atan(double x);

/**
 * std::atan() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable atan(const Variable& x);

/**
 * std::atan2() for Variables.
 *
 * @param y The y argument.
 * @param x The x argument.
 */
SLEIPNIR_DLLEXPORT Variable atan2(double y, const Variable& x);

/**
 * std::atan2() for Variables.
 *
 * @param y The y argument.
 * @param x The x argument.
 */
SLEIPNIR_DLLEXPORT Variable atan2(const Variable& y, double x);

/**
 * std::atan2() for Variables.
 *
 * @param y The y argument.
 * @param x The x argument.
 */
SLEIPNIR_DLLEXPORT Variable atan2(const Variable& y, const Variable& x);

/**
 * std::cos() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable cos(double x);

/**
 * std::cos() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable cos(const Variable& x);

/**
 * std::cosh() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable cosh(double x);

/**
 * std::cosh() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable cosh(const Variable& x);

/**
 * std::erf() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable erf(double x);

/**
 * std::erf() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable erf(const Variable& x);

/**
 * std::exp() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable exp(double x);

/**
 * std::exp() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable exp(const Variable& x);

/**
 * std::hypot() for Variables.
 *
 * @param x The x argument.
 * @param y The y argument.
 */
SLEIPNIR_DLLEXPORT Variable hypot(double x, const Variable& y);

/**
 * std::hypot() for Variables.
 *
 * @param x The x argument.
 * @param y The y argument.
 */
SLEIPNIR_DLLEXPORT Variable hypot(const Variable& x, double y);

/**
 * std::hypot() for Variables.
 *
 * @param x The x argument.
 * @param y The y argument.
 */
SLEIPNIR_DLLEXPORT Variable hypot(const Variable& x, const Variable& y);

/**
 * std::hypot() for Variables.
 *
 * @param x The x argument.
 * @param y The y argument.
 * @param z The z argument.
 */
SLEIPNIR_DLLEXPORT Variable hypot(const Variable& x, const Variable& y,
                                  const Variable& z);

/**
 * std::log() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable log(double x);

/**
 * std::log() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable log(const Variable& x);

/**
 * std::log10() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable log10(double x);

/**
 * std::log10() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable log10(const Variable& x);

/**
 * std::pow() for Variables.
 *
 * @param base The base.
 * @param power The power.
 */
SLEIPNIR_DLLEXPORT Variable pow(double base, const Variable& power);

/**
 * std::pow() for Variables.
 *
 * @param base The base.
 * @param power The power.
 */
SLEIPNIR_DLLEXPORT Variable pow(const Variable& base, double power);

/**
 * std::pow() for Variables.
 *
 * @param base The base.
 * @param power The power.
 */
SLEIPNIR_DLLEXPORT Variable pow(const Variable& base, const Variable& power);

/**
 * sign() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable sign(double x);

/**
 * sign() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable sign(const Variable& x);

/**
 * std::sin() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable sin(double x);

/**
 * std::sin() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable sin(const Variable& x);

/**
 * std::sinh() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable sinh(double x);

/**
 * std::sinh() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable sinh(const Variable& x);

/**
 * std::sqrt() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable sqrt(double x);

/**
 * std::sqrt() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable sqrt(const Variable& x);

/**
 * std::tan() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable tan(double x);

/**
 * std::tan() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable tan(const Variable& x);

/**
 * std::tanh() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable tanh(double x);

/**
 * std::tanh() for Variables.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT Variable tanh(const Variable& x);

}  // namespace sleipnir
