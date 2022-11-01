// Copyright (c) Joshua Nichols and Tyler Veness

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "sleipnir/IntrusiveSharedPtr.h"
#include "sleipnir/SymbolExports.h"
#include "sleipnir/autodiff/Expression.h"

namespace sleipnir::autodiff {

class SLEIPNIR_DLLEXPORT Variable {
 public:
  IntrusiveSharedPtr<Expression> expr;

  constexpr Variable() = default;

  Variable(const Variable&) = default;
  Variable& operator=(const Variable&) = default;

  Variable(Variable&&) = default;
  Variable& operator=(Variable&&) = default;

  Variable(double value);  // NOLINT

  Variable(int value);  // NOLINT

  /**
   * Constructs a Variable pointing to the specified expression.
   *
   * @param expr The autodiff variable.
   */
  explicit Variable(IntrusiveSharedPtr<Expression> expr);

  Variable& operator=(double value);

  Variable& operator=(int value);

  friend SLEIPNIR_DLLEXPORT Variable operator*(double lhs, const Variable& rhs);

  friend SLEIPNIR_DLLEXPORT Variable operator*(const Variable& lhs, double rhs);

  friend SLEIPNIR_DLLEXPORT Variable operator*(const Variable& lhs,
                                               const Variable& rhs);

  Variable& operator*=(double rhs);

  Variable& operator*=(const Variable& rhs);

  friend SLEIPNIR_DLLEXPORT Variable operator/(double lhs, const Variable& rhs);

  friend SLEIPNIR_DLLEXPORT Variable operator/(const Variable& lhs, double rhs);

  friend SLEIPNIR_DLLEXPORT Variable operator/(const Variable& lhs,
                                               const Variable& rhs);

  Variable& operator/=(double rhs);

  Variable& operator/=(const Variable& rhs);

  friend SLEIPNIR_DLLEXPORT Variable operator+(double lhs, const Variable& rhs);

  friend SLEIPNIR_DLLEXPORT Variable operator+(const Variable& lhs, double rhs);

  friend SLEIPNIR_DLLEXPORT Variable operator+(const Variable& lhs,
                                               const Variable& rhs);

  Variable& operator+=(double rhs);

  Variable& operator+=(const Variable& rhs);

  friend SLEIPNIR_DLLEXPORT Variable operator-(double lhs, const Variable& rhs);

  friend SLEIPNIR_DLLEXPORT Variable operator-(const Variable& lhs, double rhs);

  friend SLEIPNIR_DLLEXPORT Variable operator-(const Variable& lhs,
                                               const Variable& rhs);

  Variable& operator-=(double rhs);

  Variable& operator-=(const Variable& rhs);

  friend SLEIPNIR_DLLEXPORT Variable operator-(const Variable& lhs);

  friend SLEIPNIR_DLLEXPORT Variable operator+(const Variable& lhs);

  friend SLEIPNIR_DLLEXPORT bool operator==(double lhs, const Variable& rhs);

  friend SLEIPNIR_DLLEXPORT bool operator==(const Variable& lhs, double rhs);

  friend SLEIPNIR_DLLEXPORT bool operator==(const Variable& lhs,
                                            const Variable& rhs);

  friend SLEIPNIR_DLLEXPORT bool operator!=(double lhs, const Variable& rhs);

  friend SLEIPNIR_DLLEXPORT bool operator!=(const Variable& lhs, double rhs);

  friend SLEIPNIR_DLLEXPORT bool operator!=(const Variable& lhs,
                                            const Variable& rhs);

  friend SLEIPNIR_DLLEXPORT bool operator<(double lhs, const Variable& rhs);

  friend SLEIPNIR_DLLEXPORT bool operator<(const Variable& lhs, double rhs);

  friend SLEIPNIR_DLLEXPORT bool operator<(const Variable& lhs,
                                           const Variable& rhs);

  friend SLEIPNIR_DLLEXPORT bool operator>(double lhs, const Variable& rhs);

  friend SLEIPNIR_DLLEXPORT bool operator>(const Variable& lhs, double rhs);

  friend SLEIPNIR_DLLEXPORT bool operator>(const Variable& lhs,
                                           const Variable& rhs);

  friend SLEIPNIR_DLLEXPORT bool operator<=(double lhs, const Variable& rhs);

  friend SLEIPNIR_DLLEXPORT bool operator<=(const Variable& lhs, double rhs);

  friend SLEIPNIR_DLLEXPORT bool operator<=(const Variable& lhs,
                                            const Variable& rhs);

  friend SLEIPNIR_DLLEXPORT bool operator>=(double lhs, const Variable& rhs);

  friend SLEIPNIR_DLLEXPORT bool operator>=(const Variable& lhs, double rhs);

  friend SLEIPNIR_DLLEXPORT bool operator>=(const Variable& lhs,
                                            const Variable& rhs);

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
   * Update the value of this variable based on the values of its dependent
   * variables.
   */
  void Update();
};

using VectorXvar = Eigen::Vector<sleipnir::autodiff::Variable, Eigen::Dynamic>;
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

}  // namespace sleipnir::autodiff
