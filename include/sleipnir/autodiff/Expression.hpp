// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "sleipnir/autodiff/ExpressionType.hpp"
#include "sleipnir/util/IntrusiveSharedPtr.hpp"
#include "sleipnir/util/Pool.hpp"
#include "sleipnir/util/SymbolExports.hpp"

namespace sleipnir::detail {

struct SLEIPNIR_DLLEXPORT Expression;

/**
 * Typedef for intrusive shared pointer to Expression.
 */
using ExpressionPtr = IntrusiveSharedPtr<Expression>;

/**
 * Returns an instance of "zero", which has special meaning in expression
 * operations.
 */
SLEIPNIR_DLLEXPORT const ExpressionPtr& Zero();

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
  using TrinaryFuncExpr = ExpressionPtr (*)(const ExpressionPtr&,
                                            const ExpressionPtr&,
                                            const ExpressionPtr&);

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
  ExpressionPtr adjointExpr{Zero()};

  /// Expression argument type.
  ExpressionType type = ExpressionType::kConstant;

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
      [](const ExpressionPtr&, const ExpressionPtr&, const ExpressionPtr&) {
        return Zero();
      },
      [](const ExpressionPtr&, const ExpressionPtr&, const ExpressionPtr&) {
        return Zero();
      }};

  /// Expression arguments.
  std::array<ExpressionPtr, 2> args{Zero(), Zero()};

  /// Reference count for intrusive shared pointer.
  uint32_t refCount = 0;

  /**
   * Constructs a constant expression with a value of zero.
   */
  Expression() = default;

  /**
   * Constructs a nullary expression (an operator with no arguments).
   *
   * @param value The expression value.
   * @param type The expression type. It should be either constant (the default)
   *             or linear.
   */
  explicit Expression(double value,
                      ExpressionType type = ExpressionType::kConstant);

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
             TrinaryFuncExpr lhsGradientFunc, ExpressionPtr lhs);

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
             ExpressionPtr lhs, ExpressionPtr rhs);

  /**
   * Returns true if the expression is the given constant.
   *
   * @param constant The constant.
   */
  bool IsConstant(double constant) const;

  /**
   * Expression-Expression multiplication operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT ExpressionPtr operator*(const ExpressionPtr& lhs,
                                                    const ExpressionPtr& rhs);

  /**
   * Expression-Expression division operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT ExpressionPtr operator/(const ExpressionPtr& lhs,
                                                    const ExpressionPtr& rhs);

  /**
   * Expression-Expression addition operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT ExpressionPtr operator+(const ExpressionPtr& lhs,
                                                    const ExpressionPtr& rhs);

  /**
   * Expression-Expression subtraction operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT ExpressionPtr operator-(const ExpressionPtr& lhs,
                                                    const ExpressionPtr& rhs);

  /**
   * Unary minus operator.
   *
   * @param lhs Operand of unary minus.
   */
  friend SLEIPNIR_DLLEXPORT ExpressionPtr operator-(const ExpressionPtr& lhs);

  /**
   * Unary plus operator.
   *
   * @param lhs Operand of unary plus.
   */
  friend SLEIPNIR_DLLEXPORT ExpressionPtr operator+(const ExpressionPtr& lhs);
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
  // If a deeply nested tree is being deallocated all at once, calling the
  // Expression destructor when expr's refcount reaches zero can cause a stack
  // overflow. Instead, we iterate over its children to decrement their
  // refcounts and deallocate them.
  std::vector<Expression*> stack;
  stack.emplace_back(expr);

  while (!stack.empty()) {
    auto elem = stack.back();
    stack.pop_back();

    // Decrement the current node's refcount. If it reaches zero, deallocate the
    // node and enqueue its children so their refcounts are decremented too.
    if (--elem->refCount == 0) {
      if (elem->adjointExpr != nullptr) {
        stack.emplace_back(elem->adjointExpr.Get());
      }
      for (auto&& arg : elem->args) {
        if (arg != nullptr) {
          stack.emplace_back(arg.Get());
        }
      }

      // Not calling the destructor here is safe because it only decrements
      // refcounts, which was already done above.
      auto alloc = GlobalPoolAllocator<Expression>();
      std::allocator_traits<decltype(alloc)>::deallocate(alloc, elem,
                                                         sizeof(Expression));
    }
  }
}

/**
 * Creates an intrusive shared pointer to an expression from the global pool
 * allocator.
 *
 * @param args Constructor arguments for Expression.
 */
template <typename... Args>
static ExpressionPtr MakeExpressionPtr(Args&&... args) {
  return AllocateIntrusiveShared<Expression>(GlobalPoolAllocator<Expression>(),
                                             std::forward<Args>(args)...);
}

/**
 * std::abs() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT ExpressionPtr abs(  // NOLINT
    const ExpressionPtr& x);

/**
 * std::acos() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT ExpressionPtr acos(  // NOLINT
    const ExpressionPtr& x);

/**
 * std::asin() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT ExpressionPtr asin(  // NOLINT
    const ExpressionPtr& x);

/**
 * std::atan() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT ExpressionPtr atan(  // NOLINT
    const ExpressionPtr& x);

/**
 * std::atan2() for Expressions.
 *
 * @param y The y argument.
 * @param x The x argument.
 */
SLEIPNIR_DLLEXPORT ExpressionPtr atan2(  // NOLINT
    const ExpressionPtr& y, const ExpressionPtr& x);

/**
 * std::cos() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT ExpressionPtr cos(  // NOLINT
    const ExpressionPtr& x);

/**
 * std::cosh() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT ExpressionPtr cosh(  // NOLINT
    const ExpressionPtr& x);

/**
 * std::erf() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT ExpressionPtr erf(  // NOLINT
    const ExpressionPtr& x);

/**
 * std::exp() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT ExpressionPtr exp(  // NOLINT
    const ExpressionPtr& x);

/**
 * std::hypot() for Expressions.
 *
 * @param x The x argument.
 * @param y The y argument.
 */
SLEIPNIR_DLLEXPORT ExpressionPtr hypot(  // NOLINT
    const ExpressionPtr& x, const ExpressionPtr& y);

/**
 * std::log() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT ExpressionPtr log(  // NOLINT
    const ExpressionPtr& x);

/**
 * std::log10() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT ExpressionPtr log10(  // NOLINT
    const ExpressionPtr& x);

/**
 * std::pow() for Expressions.
 *
 * @param base The base.
 * @param power The power.
 */
SLEIPNIR_DLLEXPORT ExpressionPtr pow(  // NOLINT
    const ExpressionPtr& base, const ExpressionPtr& power);

/**
 * sign() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT ExpressionPtr sign(const ExpressionPtr& x);

/**
 * std::sin() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT ExpressionPtr sin(  // NOLINT
    const ExpressionPtr& x);

/**
 * std::sinh() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT ExpressionPtr sinh(const ExpressionPtr& x);

/**
 * std::sqrt() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT ExpressionPtr sqrt(  // NOLINT
    const ExpressionPtr& x);

/**
 * std::tan() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT ExpressionPtr tan(  // NOLINT
    const ExpressionPtr& x);

/**
 * std::tanh() for Expressions.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT ExpressionPtr tanh(const ExpressionPtr& x);

}  // namespace sleipnir::detail

namespace sleipnir {

// FIXME: Doxygen is confused:
//
//   Found ';' while parsing initializer list! (doxygen could be confused by a
//   macro call without semicolon)

//! @cond Doxygen_Suppress

// Instantiate Expression pool in Expression.cpp instead to avoid ODR violation
extern template EXPORT_TEMPLATE_DECLARE(SLEIPNIR_DLLEXPORT)
    PoolAllocator<detail::Expression> GlobalPoolAllocator<detail::Expression>();

//! @endcond

}  // namespace sleipnir
