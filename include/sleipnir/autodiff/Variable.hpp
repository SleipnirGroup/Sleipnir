// Copyright (c) Sleipnir contributors

#pragma once

#include <vector>

#include "sleipnir/autodiff/Expression.hpp"
#include "sleipnir/util/SymbolExports.hpp"

namespace sleipnir {

// Forward declarations for friend declarations in Variable
class SLEIPNIR_DLLEXPORT Jacobian;
namespace detail {
class SLEIPNIR_DLLEXPORT ExpressionGraph;
}  // namespace detail

/**
 * An autodiff variable pointing to an expression node.
 */
class SLEIPNIR_DLLEXPORT Variable {
 public:
  /**
   * Constructs a linear Variable with a value of zero.
   */
  Variable() = default;

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
  explicit Variable(const detail::ExpressionPtr& expr);

  /**
   * Constructs a Variable pointing to the specified expression.
   *
   * @param expr The autodiff variable.
   */
  explicit Variable(detail::ExpressionPtr&& expr);

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
   * Variable-Variable multiplication operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT Variable operator*(const Variable& lhs,
                                               const Variable& rhs);

  /**
   * Variable-Variable compound multiplication operator.
   *
   * @param rhs Operator right-hand side.
   */
  Variable& operator*=(const Variable& rhs);

  /**
   * Variable-Variable division operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT Variable operator/(const Variable& lhs,
                                               const Variable& rhs);

  /**
   * Variable-Variable compound division operator.
   *
   * @param rhs Operator right-hand side.
   */
  Variable& operator/=(const Variable& rhs);

  /**
   * Variable-Variable addition operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT Variable operator+(const Variable& lhs,
                                               const Variable& rhs);

  /**
   * Variable-Variable compound addition operator.
   *
   * @param rhs Operator right-hand side.
   */
  Variable& operator+=(const Variable& rhs);

  /**
   * Variable-Variable subtraction operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT Variable operator-(const Variable& lhs,
                                               const Variable& rhs);

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

 private:
  /// The expression node.
  detail::ExpressionPtr expr =
      detail::MakeExpressionPtr(0.0, ExpressionType::kLinear);

  friend SLEIPNIR_DLLEXPORT Variable abs(const Variable& x);
  friend SLEIPNIR_DLLEXPORT Variable acos(const Variable& x);
  friend SLEIPNIR_DLLEXPORT Variable asin(const Variable& x);
  friend SLEIPNIR_DLLEXPORT Variable atan(const Variable& x);
  friend SLEIPNIR_DLLEXPORT Variable atan2(const Variable& y,
                                           const Variable& x);
  friend SLEIPNIR_DLLEXPORT Variable cos(const Variable& x);
  friend SLEIPNIR_DLLEXPORT Variable cosh(const Variable& x);
  friend SLEIPNIR_DLLEXPORT Variable erf(const Variable& x);
  friend SLEIPNIR_DLLEXPORT Variable exp(const Variable& x);
  friend SLEIPNIR_DLLEXPORT Variable hypot(const Variable& x,
                                           const Variable& y);
  friend SLEIPNIR_DLLEXPORT Variable hypot(const Variable& x, const Variable& y,
                                           const Variable& z);
  friend SLEIPNIR_DLLEXPORT Variable log(const Variable& x);
  friend SLEIPNIR_DLLEXPORT Variable log10(const Variable& x);
  friend SLEIPNIR_DLLEXPORT Variable pow(const Variable& base,
                                         const Variable& power);
  friend SLEIPNIR_DLLEXPORT Variable sign(const Variable& x);
  friend SLEIPNIR_DLLEXPORT Variable sin(const Variable& x);
  friend SLEIPNIR_DLLEXPORT Variable sinh(const Variable& x);
  friend SLEIPNIR_DLLEXPORT Variable sqrt(const Variable& x);
  friend SLEIPNIR_DLLEXPORT Variable tan(const Variable& x);
  friend SLEIPNIR_DLLEXPORT Variable tanh(const Variable& x);

  friend class SLEIPNIR_DLLEXPORT Jacobian;

  // FIXME: Doxygen is confused:
  //
  //   member 'SLEIPNIR_DLLEXPORT detail::ExpressionGraph' of class 'Variable'
  //   cannot be found

  //! @cond Doxygen_Suppress
  friend class SLEIPNIR_DLLEXPORT detail::ExpressionGraph;
  //! @endcond
};

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
SLEIPNIR_DLLEXPORT Variable acos(const Variable& x);

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
SLEIPNIR_DLLEXPORT Variable atan(const Variable& x);

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
SLEIPNIR_DLLEXPORT Variable cos(const Variable& x);

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
SLEIPNIR_DLLEXPORT Variable erf(const Variable& x);

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
SLEIPNIR_DLLEXPORT Variable log(const Variable& x);

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
SLEIPNIR_DLLEXPORT Variable pow(const Variable& base, const Variable& power);

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
SLEIPNIR_DLLEXPORT Variable sin(const Variable& x);

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
SLEIPNIR_DLLEXPORT Variable sqrt(const Variable& x);

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
SLEIPNIR_DLLEXPORT Variable tanh(const Variable& x);

}  // namespace sleipnir
