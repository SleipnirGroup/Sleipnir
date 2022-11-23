// Copyright (c) Joshua Nichols and Tyler Veness

#pragma once

#include <stdint.h>

#include <array>
#include <memory>

#include "sleipnir/IntrusiveSharedPtr.hpp"
#include "sleipnir/SymbolExports.hpp"
#include "sleipnir/autodiff/Pool.hpp"

namespace sleipnir::autodiff {

enum class ExpressionType { kNone, kConstant, kLinear, kQuadratic, kNonlinear };

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
  using TrinaryFuncExpr = sleipnir::IntrusiveSharedPtr<Expression> (*)(
      const sleipnir::IntrusiveSharedPtr<Expression>&,
      const sleipnir::IntrusiveSharedPtr<Expression>&,
      const sleipnir::IntrusiveSharedPtr<Expression>&);

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

  /// The expression's creation order. The value assigned here is from a
  /// monotonically increasing counter that increments every time an Expression
  /// is constructed. This is used for sorting a flattened representation of the
  /// expression tree in autodiff Jacobian or Hessian.
  size_t id;

  /// The adjoint of the expression node used during gradient expression tree
  /// generation.
  sleipnir::IntrusiveSharedPtr<Expression> adjointExpr;

  /// Expression argument type.
  ExpressionType type = sleipnir::autodiff::ExpressionType::kLinear;

  /// Either nullary operator with no arguments, unary operator with one
  /// argument, or binary operator with two arguments. This operator is
  /// used to update the node's value.
  BinaryFuncDouble valueFunc = [](double, double) { return 0.0; };

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
      [](const sleipnir::IntrusiveSharedPtr<Expression>&,
         const sleipnir::IntrusiveSharedPtr<Expression>&,
         const sleipnir::IntrusiveSharedPtr<Expression>&) {
        return sleipnir::IntrusiveSharedPtr<Expression>{};
      },
      [](const sleipnir::IntrusiveSharedPtr<Expression>&,
         const sleipnir::IntrusiveSharedPtr<Expression>&,
         const sleipnir::IntrusiveSharedPtr<Expression>&) {
        return sleipnir::IntrusiveSharedPtr<Expression>{};
      }};

  /// Expression arguments.
  std::array<sleipnir::IntrusiveSharedPtr<Expression>, 2> args{nullptr,
                                                               nullptr};

  /// Reference count for intrusive shared pointer.
  uint32_t refCount = 0;

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
             sleipnir::IntrusiveSharedPtr<Expression> lhs);

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
             sleipnir::IntrusiveSharedPtr<Expression> lhs,
             sleipnir::IntrusiveSharedPtr<Expression> rhs);

  /**
   * Double-expression multiplication operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> operator*(
      double lhs, const sleipnir::IntrusiveSharedPtr<Expression>& rhs);

  /**
   * Expression-double multiplication operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> operator*(
      const sleipnir::IntrusiveSharedPtr<Expression>& lhs, double rhs);

  /**
   * Expression-Expression multiplication operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> operator*(
      const sleipnir::IntrusiveSharedPtr<Expression>& lhs,
      const sleipnir::IntrusiveSharedPtr<Expression>& rhs);

  /**
   * double-Expression division operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> operator/(
      double lhs, const sleipnir::IntrusiveSharedPtr<Expression>& rhs);

  /**
   * Expression-double division operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> operator/(
      const sleipnir::IntrusiveSharedPtr<Expression>& lhs, double rhs);

  /**
   * Expression-Expression division operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> operator/(
      const sleipnir::IntrusiveSharedPtr<Expression>& lhs,
      const sleipnir::IntrusiveSharedPtr<Expression>& rhs);

  /**
   * double-Expression addition operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> operator+(
      double lhs, const sleipnir::IntrusiveSharedPtr<Expression>& rhs);

  /**
   * Expression-double addition operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> operator+(
      const sleipnir::IntrusiveSharedPtr<Expression>& lhs, double rhs);

  /**
   * Expression-Expression addition operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> operator+(
      const sleipnir::IntrusiveSharedPtr<Expression>& lhs,
      const sleipnir::IntrusiveSharedPtr<Expression>& rhs);

  /**
   * double-Expression subtraction operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> operator-(
      double lhs, const sleipnir::IntrusiveSharedPtr<Expression>& rhs);

  /**
   * Expression-double subtraction operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> operator-(
      const sleipnir::IntrusiveSharedPtr<Expression>& lhs, double rhs);

  /**
   * Expression-Expression subtraction operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> operator-(
      const sleipnir::IntrusiveSharedPtr<Expression>& lhs,
      const sleipnir::IntrusiveSharedPtr<Expression>& rhs);

  /**
   * Unary minus operator.
   *
   * @param lhs Operand of unary minus.
   */
  friend SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> operator-(
      const sleipnir::IntrusiveSharedPtr<Expression>& lhs);

  /**
   * Unary plus operator.
   *
   * @param lhs Operand of unary plus.
   */
  friend SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> operator+(
      const sleipnir::IntrusiveSharedPtr<Expression>& lhs);
};

/**
 * Returns an allocator for the global Expression pool memory resource.
 */
SLEIPNIR_DLLEXPORT PoolAllocator<Expression> Allocator();

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
    auto alloc = Allocator();
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
SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> MakeConstant(
    double x);

/**
 * std::abs() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> abs(  // NOLINT
    const sleipnir::IntrusiveSharedPtr<Expression>& x);

/**
 * std::acos() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> acos(  // NOLINT
    const sleipnir::IntrusiveSharedPtr<Expression>& x);

/**
 * std::asin() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> asin(  // NOLINT
    const sleipnir::IntrusiveSharedPtr<Expression>& x);

/**
 * std::atan() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> atan(  // NOLINT
    const sleipnir::IntrusiveSharedPtr<Expression>& x);

/**
 * std::atan2() for Expressions.
 *
 * @param y The y argument.
 * @param x The x argument.
 */
SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> atan2(  // NOLINT
    const sleipnir::IntrusiveSharedPtr<Expression>& y,
    const sleipnir::IntrusiveSharedPtr<Expression>& x);

/**
 * std::cos() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> cos(  // NOLINT
    const sleipnir::IntrusiveSharedPtr<Expression>& x);

/**
 * std::cosh() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> cosh(  // NOLINT
    const sleipnir::IntrusiveSharedPtr<Expression>& x);

/**
 * std::erf() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> erf(  // NOLINT
    const sleipnir::IntrusiveSharedPtr<Expression>& x);

/**
 * std::exp() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> exp(  // NOLINT
    const sleipnir::IntrusiveSharedPtr<Expression>& x);

/**
 * std::hypot() for Expressions.
 *
 * @param x The x argument.
 * @param y The y argument.
 */
SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> hypot(  // NOLINT
    const sleipnir::IntrusiveSharedPtr<Expression>& x,
    const sleipnir::IntrusiveSharedPtr<Expression>& y);

/**
 * std::log() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> log(  // NOLINT
    const sleipnir::IntrusiveSharedPtr<Expression>& x);

/**
 * std::log10() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> log10(  // NOLINT
    const sleipnir::IntrusiveSharedPtr<Expression>& x);

/**
 * std::pow() for Expressions.
 *
 * @param base The base.
 * @param power The power.
 */
SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> pow(  // NOLINT
    const sleipnir::IntrusiveSharedPtr<Expression>& base,
    const sleipnir::IntrusiveSharedPtr<Expression>& power);

/**
 * std::sin() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> sin(  // NOLINT
    const sleipnir::IntrusiveSharedPtr<Expression>& x);

/**
 * std::sinh() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> sinh(
    const sleipnir::IntrusiveSharedPtr<Expression>& x);

/**
 * std::sqrt() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> sqrt(  // NOLINT
    const sleipnir::IntrusiveSharedPtr<Expression>& x);

/**
 * std::tan() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> tan(  // NOLINT
    const sleipnir::IntrusiveSharedPtr<Expression>& x);

/**
 * std::tanh() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT sleipnir::IntrusiveSharedPtr<Expression> tanh(
    const sleipnir::IntrusiveSharedPtr<Expression>& x);

}  // namespace sleipnir::autodiff
