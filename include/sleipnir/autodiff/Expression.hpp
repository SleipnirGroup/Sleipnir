// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <array>
#include <memory>

#include "sleipnir/IntrusiveSharedPtr.hpp"
#include "sleipnir/Pool.hpp"
#include "sleipnir/SymbolExports.hpp"

namespace sleipnir {

enum class ExpressionType { kNone, kConstant, kLinear, kQuadratic, kNonlinear };

struct SLEIPNIR_DLLEXPORT Expression;

SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression>& Zero();

/**
 * An autodiff expression node.
 */
struct SLEIPNIR_DLLEXPORT Expression {
  /**
   * Binary function taking two doubles and returning a double.
   */
  using BinaryFuncDouble = double (*)(double, double);

  /**
   * Trinary function taking three doubles and returning a double.
   */
  using TrinaryFuncDouble = double (*)(double, double, double);

  /**
   * Trinary function taking three expressions and returning an expression.
   */
  using TrinaryFuncExpr =
      IntrusiveSharedPtr<Expression> (*)(const IntrusiveSharedPtr<Expression>&,
                                         const IntrusiveSharedPtr<Expression>&,
                                         const IntrusiveSharedPtr<Expression>&);

  /// The value of the expression node.
  double value = 0.0;

  /// The adjoint of the expression node used during autodiff.
  double adjoint = 0.0;

  /// Tracks the number of instances of this expression yet to be encountered in
  /// an expression tree.
  int duplications = 0;

  /// This expression's row in wrt for autodiff gradient, Jacobian, or Hessian.
  /// This is -1 if the expression isn't in wrt.
  int row = -1;

  /// The adjoint of the expression node used during gradient expression tree
  /// generation.
  IntrusiveSharedPtr<Expression> adjointExpr;

  /// Expression argument type.
  ExpressionType type = ExpressionType::kLinear;

  /// Either nullary operator with no arguments, unary operator with one
  /// argument, or binary operator with two arguments. This operator is
  /// used to update the node's value.
  BinaryFuncDouble valueFunc = nullptr;

  /// Functions returning double adjoints of the children expressions.
  ///
  /// Parameters:
  /// <ul>
  ///   <li>lhs: Left argument to binary operator.</li>
  ///   <li>rhs: Right argument to binary operator.</li>
  ///   <li>parentAdjoint: Adjoint of parent expression.</li>
  /// </ul>
  std::array<TrinaryFuncDouble, 2> gradientValueFuncs{
      [](double, double, double) { return 0.0; },
      [](double, double, double) { return 0.0; }};

  /// Functions returning Variable adjoints of the children expressions.
  ///
  /// Parameters:
  /// <ul>
  ///   <li>lhs: Left argument to binary operator.</li>
  ///   <li>rhs: Right argument to binary operator.</li>
  ///   <li>parentAdjoint: Adjoint of parent expression.</li>
  /// </ul>
  std::array<TrinaryFuncExpr, 2> gradientFuncs{
      [](const IntrusiveSharedPtr<Expression>&,
         const IntrusiveSharedPtr<Expression>&,
         const IntrusiveSharedPtr<Expression>&) { return Zero(); },
      [](const IntrusiveSharedPtr<Expression>&,
         const IntrusiveSharedPtr<Expression>&,
         const IntrusiveSharedPtr<Expression>&) { return Zero(); }};

  /// Expression arguments.
  std::array<IntrusiveSharedPtr<Expression>, 2> args;

  /// Reference count for intrusive shared pointer.
  uint32_t refCount = 0;

  /**
   * Type tag used for constructing the "zero" expression.
   */
  struct ZeroSingleton_t {};

  /**
   * Type tag used for constructing the "zero" expression.
   */
  static inline ZeroSingleton_t ZeroSingleton;

  /**
   * Constructs an instance of "zero", which has special meaning in expression
   * operations. This should only be constructed once via Zero().
   */
  explicit Expression(ZeroSingleton_t);

  /**
   * Copy constructor.
   */
  Expression(const Expression&) = default;

  /**
   * Copy-assignment operator.
   */
  Expression& operator=(const Expression&) = default;

  /**
   * Move constructor.
   */
  Expression(Expression&&) = default;

  /**
   * Move-assignment operator.
   */
  Expression& operator=(Expression&&) = default;

  /**
   * Constructs a nullary expression (an operator with no arguments).
   *
   * @param value The expression value.
   * @param type The expression type. It should be either linear (the default)
   *             or constant.
   */
  explicit Expression(double value,
                      ExpressionType type = ExpressionType::kLinear);

  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param type The expression's type.
   * @param valueFunc Unary operator that produces this expression's value.
   * @param lhsGradientValueFunc Gradient with respect to the operand.
   * @param lhsGradientFunc Gradient with respect to the operand.
   * @param lhs Unary operator's operand.
   */
  Expression(ExpressionType type, BinaryFuncDouble valueFunc,
             TrinaryFuncDouble lhsGradientValueFunc,
             TrinaryFuncExpr lhsGradientFunc,
             IntrusiveSharedPtr<Expression> lhs);

  /**
   * Constructs a binary expression (an operator with two arguments).
   *
   * @param type The expression's type.
   * @param valueFunc Unary operator that produces this expression's value.
   * @param lhsGradientValueFunc Gradient with respect to the left operand.
   * @param rhsGradientValueFunc Gradient with respect to the right operand.
   * @param lhsGradientFunc Gradient with respect to the left operand.
   * @param rhsGradientFunc Gradient with respect to the right operand.
   * @param lhs Binary operator's left operand.
   * @param rhs Binary operator's right operand.
   */
  Expression(ExpressionType type, BinaryFuncDouble valueFunc,
             TrinaryFuncDouble lhsGradientValueFunc,
             TrinaryFuncDouble rhsGradientValueFunc,
             TrinaryFuncExpr lhsGradientFunc, TrinaryFuncExpr rhsGradientFunc,
             IntrusiveSharedPtr<Expression> lhs,
             IntrusiveSharedPtr<Expression> rhs);

  /**
   * Double-expression multiplication operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator*(
      double lhs, const IntrusiveSharedPtr<Expression>& rhs);

  /**
   * Expression-double multiplication operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator*(
      const IntrusiveSharedPtr<Expression>& lhs, double rhs);

  /**
   * Expression-Expression multiplication operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator*(
      const IntrusiveSharedPtr<Expression>& lhs,
      const IntrusiveSharedPtr<Expression>& rhs);

  /**
   * double-Expression division operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator/(
      double lhs, const IntrusiveSharedPtr<Expression>& rhs);

  /**
   * Expression-double division operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator/(
      const IntrusiveSharedPtr<Expression>& lhs, double rhs);

  /**
   * Expression-Expression division operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator/(
      const IntrusiveSharedPtr<Expression>& lhs,
      const IntrusiveSharedPtr<Expression>& rhs);

  /**
   * double-Expression addition operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator+(
      double lhs, const IntrusiveSharedPtr<Expression>& rhs);

  /**
   * Expression-double addition operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator+(
      const IntrusiveSharedPtr<Expression>& lhs, double rhs);

  /**
   * Expression-Expression addition operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator+(
      const IntrusiveSharedPtr<Expression>& lhs,
      const IntrusiveSharedPtr<Expression>& rhs);

  /**
   * Expression-double compound addition operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression>& operator+=(
      IntrusiveSharedPtr<Expression>& lhs, double rhs);

  /**
   * Expression-Expression compound addition operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression>& operator+=(
      IntrusiveSharedPtr<Expression>& lhs,
      const IntrusiveSharedPtr<Expression>& rhs);

  /**
   * double-Expression subtraction operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator-(
      double lhs, const IntrusiveSharedPtr<Expression>& rhs);

  /**
   * Expression-double subtraction operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator-(
      const IntrusiveSharedPtr<Expression>& lhs, double rhs);

  /**
   * Expression-Expression subtraction operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator-(
      const IntrusiveSharedPtr<Expression>& lhs,
      const IntrusiveSharedPtr<Expression>& rhs);

  /**
   * Unary minus operator.
   *
   * @param lhs Operand of unary minus.
   */
  friend SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator-(
      const IntrusiveSharedPtr<Expression>& lhs);

  /**
   * Unary plus operator.
   *
   * @param lhs Operand of unary plus.
   */
  friend SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> operator+(
      const IntrusiveSharedPtr<Expression>& lhs);
};

/**
 * Refcount increment for intrusive shared pointer.
 *
 * @param expr The shared pointer's managed object.
 */
inline void IntrusiveSharedPtrIncRefCount(Expression* expr) {
  ++expr->refCount;
}

/**
 * Refcount decrement for intrusive shared pointer.
 *
 * @param expr The shared pointer's managed object.
 */
inline void IntrusiveSharedPtrDecRefCount(Expression* expr) {
  if (--expr->refCount == 0) {
    auto alloc = GlobalPoolAllocator<Expression>();
    std::allocator_traits<decltype(alloc)>::destroy(alloc, expr);
    std::allocator_traits<decltype(alloc)>::deallocate(alloc, expr,
                                                       sizeof(Expression));
  }
}

/**
 * Creates a constant expression.
 *
 * @param x The constant.
 */
SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> MakeConstant(double x);

/**
 * std::abs() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> abs(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x);

/**
 * std::acos() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> acos(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x);

/**
 * std::asin() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> asin(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x);

/**
 * std::atan() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> atan(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x);

/**
 * std::atan2() for Expressions.
 *
 * @param y The y argument.
 * @param x The x argument.
 */
SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> atan2(  // NOLINT
    const IntrusiveSharedPtr<Expression>& y,
    const IntrusiveSharedPtr<Expression>& x);

/**
 * std::cos() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> cos(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x);

/**
 * std::cosh() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> cosh(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x);

/**
 * std::erf() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> erf(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x);

/**
 * std::exp() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> exp(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x);

/**
 * std::hypot() for Expressions.
 *
 * @param x The x argument.
 * @param y The y argument.
 */
SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> hypot(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x,
    const IntrusiveSharedPtr<Expression>& y);

/**
 * std::log() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> log(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x);

/**
 * std::log10() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> log10(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x);

/**
 * std::pow() for Expressions.
 *
 * @param base The base.
 * @param power The power.
 */
SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> pow(  // NOLINT
    const IntrusiveSharedPtr<Expression>& base,
    const IntrusiveSharedPtr<Expression>& power);

/**
 * sign() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> sign(
    const IntrusiveSharedPtr<Expression>& x);

/**
 * std::sin() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> sin(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x);

/**
 * std::sinh() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> sinh(
    const IntrusiveSharedPtr<Expression>& x);

/**
 * std::sqrt() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> sqrt(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x);

/**
 * std::tan() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> tan(  // NOLINT
    const IntrusiveSharedPtr<Expression>& x);

/**
 * std::tanh() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT IntrusiveSharedPtr<Expression> tanh(
    const IntrusiveSharedPtr<Expression>& x);

}  // namespace sleipnir
