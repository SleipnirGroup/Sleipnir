// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <numbers>
#include <utility>

#include "sleipnir/autodiff/ExpressionType.hpp"
#include "sleipnir/util/IntrusiveSharedPtr.hpp"
#include "sleipnir/util/Pool.hpp"
#include "sleipnir/util/small_vector.hpp"

namespace sleipnir::detail {

// The global pool allocator uses a thread-local static pool resource, which
// isn't guaranteed to be initialized properly across DLL boundaries on Windows
#ifdef _WIN32
inline constexpr bool kUsePoolAllocator = false;
#else
inline constexpr bool kUsePoolAllocator = true;
#endif

struct Expression;

inline constexpr void IntrusiveSharedPtrIncRefCount(Expression* expr);
inline constexpr void IntrusiveSharedPtrDecRefCount(Expression* expr);

/**
 * Typedef for intrusive shared pointer to Expression.
 */
using ExpressionPtr = IntrusiveSharedPtr<Expression>;

/**
 * Creates an intrusive shared pointer to an expression from the global pool
 * allocator.
 *
 * @tparam T The derived expression type.
 * @param args Constructor arguments for Expression.
 */
template <typename T, typename... Args>
static ExpressionPtr MakeExpressionPtr(Args&&... args) {
  if constexpr (kUsePoolAllocator) {
    return AllocateIntrusiveShared<T>(GlobalPoolAllocator<T>(),
                                      std::forward<Args>(args)...);
  } else {
    return MakeIntrusiveShared<T>(std::forward<Args>(args)...);
  }
}

struct BinaryMinusExpression;
struct BinaryPlusExpression;
struct ConstExpression;
struct DivExpression;
struct MultExpression;
struct UnaryMinusExpression;

/**
 * An autodiff expression node.
 */
struct Expression {
  /// The value of the expression node.
  double value = 0.0;

  /// The adjoint of the expression node used during autodiff.
  double adjoint = 0.0;

  /// Tracks the number of instances of this expression yet to be encountered in
  /// an expression tree.
  uint32_t duplications = 0;

  /// This expression's row in wrt for autodiff gradient, Jacobian, or Hessian.
  /// This is -1 if the expression isn't in wrt.
  int32_t row = -1;

  /// The adjoint of the expression node used during gradient expression tree
  /// generation.
  ExpressionPtr adjointExpr;

  /// Expression argument type.
  ExpressionType type = ExpressionType::kConstant;

  /// Reference count for intrusive shared pointer.
  uint32_t refCount = 0;

  /// Expression arguments.
  std::array<ExpressionPtr, 2> args{nullptr, nullptr};

  /**
   * Constructs a constant expression with a value of zero.
   */
  constexpr Expression() = default;

  /**
   * Constructs a nullary expression (an operator with no arguments).
   *
   * @param value The expression value.
   * @param type The expression type. It should be either constant (the default)
   *             or linear.
   */
  explicit constexpr Expression(double value,
                                ExpressionType type = ExpressionType::kConstant)
      : value{value}, type{type} {}

  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param type The expression's type.
   * @param lhs Unary operator's operand.
   */
  constexpr Expression(ExpressionType type, ExpressionPtr lhs)
      : type{type}, args{std::move(lhs), nullptr} {}

  /**
   * Constructs a binary expression (an operator with two arguments).
   *
   * @param type The expression's type.
   * @param lhs Binary operator's left operand.
   * @param rhs Binary operator's right operand.
   */
  constexpr Expression(ExpressionType type, ExpressionPtr lhs,
                       ExpressionPtr rhs)
      : type{type}, args{std::move(lhs), std::move(rhs)} {}

  virtual ~Expression() = default;

  /**
   * Returns true if the expression is the given constant.
   *
   * @param constant The constant.
   */
  constexpr bool IsConstant(double constant) const {
    return type == ExpressionType::kConstant && value == constant;
  }

  /**
   * Expression-Expression multiplication operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend ExpressionPtr operator*(const ExpressionPtr& lhs,
                                 const ExpressionPtr& rhs) {
    using enum ExpressionType;

    // Prune expression
    if (lhs->IsConstant(0.0)) {
      // Return zero
      return lhs;
    } else if (rhs->IsConstant(0.0)) {
      // Return zero
      return rhs;
    } else if (lhs->IsConstant(1.0)) {
      return rhs;
    } else if (rhs->IsConstant(1.0)) {
      return lhs;
    }

    // Evaluate constant
    if (lhs->type == kConstant && rhs->type == kConstant) {
      return MakeExpressionPtr<ConstExpression>(lhs->value * rhs->value);
    }

    // Evaluate expression type
    ExpressionType type;
    if (lhs->type == kConstant) {
      type = rhs->type;
    } else if (rhs->type == kConstant) {
      type = lhs->type;
    } else if (lhs->type == kLinear && rhs->type == kLinear) {
      type = kQuadratic;
    } else {
      type = kNonlinear;
    }

    return MakeExpressionPtr<MultExpression>(type, lhs, rhs);
  }

  /**
   * Expression-Expression division operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend ExpressionPtr operator/(const ExpressionPtr& lhs,
                                 const ExpressionPtr& rhs) {
    using enum ExpressionType;

    // Prune expression
    if (lhs->IsConstant(0.0)) {
      // Return zero
      return lhs;
    } else if (rhs->IsConstant(1.0)) {
      return lhs;
    }

    // Evaluate constant
    if (lhs->type == kConstant && rhs->type == kConstant) {
      return MakeExpressionPtr<ConstExpression>(lhs->value / rhs->value);
    }

    // Evaluate expression type
    ExpressionType type;
    if (rhs->type == kConstant) {
      type = lhs->type;
    } else {
      type = kNonlinear;
    }

    return MakeExpressionPtr<DivExpression>(type, lhs, rhs);
  }

  /**
   * Expression-Expression addition operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend ExpressionPtr operator+(const ExpressionPtr& lhs,
                                 const ExpressionPtr& rhs) {
    using enum ExpressionType;

    // Prune expression
    if (lhs == nullptr || lhs->IsConstant(0.0)) {
      return rhs;
    } else if (rhs == nullptr || rhs->IsConstant(0.0)) {
      return lhs;
    }

    // Evaluate constant
    if (lhs->type == kConstant && rhs->type == kConstant) {
      return MakeExpressionPtr<ConstExpression>(lhs->value + rhs->value);
    }

    return MakeExpressionPtr<BinaryPlusExpression>(
        std::max(lhs->type, rhs->type), lhs, rhs);
  }

  /**
   * Expression-Expression subtraction operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend ExpressionPtr operator-(const ExpressionPtr& lhs,
                                 const ExpressionPtr& rhs) {
    using enum ExpressionType;

    // Prune expression
    if (lhs->IsConstant(0.0)) {
      if (rhs->IsConstant(0.0)) {
        // Return zero
        return rhs;
      } else {
        return -rhs;
      }
    } else if (rhs->IsConstant(0.0)) {
      return lhs;
    }

    // Evaluate constant
    if (lhs->type == kConstant && rhs->type == kConstant) {
      return MakeExpressionPtr<ConstExpression>(lhs->value - rhs->value);
    }

    return MakeExpressionPtr<BinaryMinusExpression>(
        std::max(lhs->type, rhs->type), lhs, rhs);
  }

  /**
   * Unary minus operator.
   *
   * @param lhs Operand of unary minus.
   */
  friend ExpressionPtr operator-(const ExpressionPtr& lhs) {
    using enum ExpressionType;

    // Prune expression
    if (lhs->IsConstant(0.0)) {
      // Return zero
      return lhs;
    }

    // Evaluate constant
    if (lhs->type == kConstant) {
      return MakeExpressionPtr<ConstExpression>(-lhs->value);
    }

    return MakeExpressionPtr<UnaryMinusExpression>(lhs->type, lhs);
  }

  /**
   * Unary plus operator.
   *
   * @param lhs Operand of unary plus.
   */
  friend ExpressionPtr operator+(const ExpressionPtr& lhs) { return lhs; }

  /**
   * Either nullary operator with no arguments, unary operator with one
   * argument, or binary operator with two arguments. This operator is used to
   * update the node's value.
   *
   * @param lhs Left argument to binary operator.
   * @param rhs Right argument to binary operator.
   */
  virtual double Value([[maybe_unused]] double lhs,
                       [[maybe_unused]] double rhs) const = 0;

  /**
   * Returns double adjoint of the left child expression.
   *
   * @param lhs Left argument to binary operator.
   * @param rhs Right argument to binary operator.
   * @param parentAdjoint Adjoint of parent expression.
   */
  virtual double GradientValueLhs([[maybe_unused]] double lhs,
                                  [[maybe_unused]] double rhs,
                                  double parentAdjoint) const = 0;

  /**
   * Returns double adjoint of the right child expression.
   *
   * @param lhs Left argument to binary operator.
   * @param rhs Right argument to binary operator.
   * @param parentAdjoint Adjoint of parent expression.
   */
  virtual double GradientValueRhs([[maybe_unused]] double lhs,
                                  [[maybe_unused]] double rhs,
                                  [[maybe_unused]] double parentAdjoint) const {
    return 0.0;
  }

  /**
   * Returns Variable adjoint of the left child expression.
   *
   * @param lhs Left argument to binary operator.
   * @param rhs Right argument to binary operator.
   * @param parentAdjoint Adjoint of parent expression.
   */
  virtual ExpressionPtr GradientLhs(
      [[maybe_unused]] const ExpressionPtr& lhs,
      [[maybe_unused]] const ExpressionPtr& rhs,
      const ExpressionPtr& parentAdjoint) const = 0;

  /**
   * Returns Variable adjoint of the right child expression.
   *
   * @param lhs Left argument to binary operator.
   * @param rhs Right argument to binary operator.
   * @param parentAdjoint Adjoint of parent expression.
   */
  virtual ExpressionPtr GradientRhs(
      [[maybe_unused]] const ExpressionPtr& lhs,
      [[maybe_unused]] const ExpressionPtr& rhs,
      [[maybe_unused]] const ExpressionPtr& parentAdjoint) const {
    return ExpressionPtr{nullptr};
  }
};

struct BinaryMinusExpression final : Expression {
  /**
   * Constructs a binary expression (an operator with two arguments).
   *
   * @param type The expression's type.
   * @param lhs Binary operator's left operand.
   * @param rhs Binary operator's right operand.
   */
  constexpr BinaryMinusExpression(ExpressionType type, ExpressionPtr lhs,
                                  ExpressionPtr rhs)
      : Expression{type, std::move(lhs), std::move(rhs)} {
    value = BinaryMinusExpression::Value(args[0]->value, args[1]->value);
  }

  double Value(double lhs, double rhs) const override { return lhs - rhs; }

  double GradientValueLhs(double, double, double parentAdjoint) const override {
    return parentAdjoint;
  }

  double GradientValueRhs(double, double, double parentAdjoint) const override {
    return -parentAdjoint;
  }

  ExpressionPtr GradientLhs(const ExpressionPtr&, const ExpressionPtr&,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint;
  }

  ExpressionPtr GradientRhs(const ExpressionPtr&, const ExpressionPtr&,
                            const ExpressionPtr& parentAdjoint) const override {
    return -parentAdjoint;
  }
};

struct BinaryPlusExpression final : Expression {
  /**
   * Constructs a binary expression (an operator with two arguments).
   *
   * @param type The expression's type.
   * @param lhs Binary operator's left operand.
   * @param rhs Binary operator's right operand.
   */
  constexpr BinaryPlusExpression(ExpressionType type, ExpressionPtr lhs,
                                 ExpressionPtr rhs)
      : Expression{type, std::move(lhs), std::move(rhs)} {
    value = BinaryPlusExpression::Value(args[0]->value, args[1]->value);
  }

  double Value(double lhs, double rhs) const override { return lhs + rhs; }

  double GradientValueLhs(double, double, double parentAdjoint) const override {
    return parentAdjoint;
  }

  double GradientValueRhs(double, double, double parentAdjoint) const override {
    return parentAdjoint;
  }

  ExpressionPtr GradientLhs(const ExpressionPtr&, const ExpressionPtr&,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint;
  }

  ExpressionPtr GradientRhs(const ExpressionPtr&, const ExpressionPtr&,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint;
  }
};

struct ConstExpression final : Expression {
  /**
   * Constructs a constant expression with a value of zero.
   */
  constexpr ConstExpression() = default;

  /**
   * Constructs a nullary expression (an operator with no arguments).
   *
   * @param value The expression value.
   * @param type The expression type. It should be either constant (the default)
   *             or linear.
   */
  explicit constexpr ConstExpression(
      double value, ExpressionType type = ExpressionType::kConstant)
      : Expression{value, type} {}

  double Value(double, double) const override { return value; }

  double GradientValueLhs(double, double, double) const override { return 0.0; }

  double GradientValueRhs(double, double, double) const override { return 0.0; }

  ExpressionPtr GradientLhs(const ExpressionPtr&, const ExpressionPtr&,
                            const ExpressionPtr&) const override {
    return ExpressionPtr{nullptr};
  }

  ExpressionPtr GradientRhs(const ExpressionPtr&, const ExpressionPtr&,
                            const ExpressionPtr&) const override {
    return ExpressionPtr{nullptr};
  }
};

struct DivExpression final : Expression {
  /**
   * Constructs a binary expression (an operator with two arguments).
   *
   * @param type The expression's type.
   * @param lhs Binary operator's left operand.
   * @param rhs Binary operator's right operand.
   */
  constexpr DivExpression(ExpressionType type, ExpressionPtr lhs,
                          ExpressionPtr rhs)
      : Expression{type, std::move(lhs), std::move(rhs)} {
    value = DivExpression::Value(args[0]->value, args[1]->value);
  }

  double Value(double lhs, double rhs) const override { return lhs / rhs; }

  double GradientValueLhs(double, double rhs,
                          double parentAdjoint) const override {
    return parentAdjoint / rhs;
  };

  double GradientValueRhs(double lhs, double rhs,
                          double parentAdjoint) const override {
    return parentAdjoint * -lhs / (rhs * rhs);
  }

  ExpressionPtr GradientLhs(const ExpressionPtr&, const ExpressionPtr& rhs,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint / rhs;
  }

  ExpressionPtr GradientRhs(const ExpressionPtr& lhs, const ExpressionPtr& rhs,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * -lhs / (rhs * rhs);
  }
};

struct MultExpression final : Expression {
  /**
   * Constructs a binary expression (an operator with two arguments).
   *
   * @param type The expression's type.
   * @param lhs Binary operator's left operand.
   * @param rhs Binary operator's right operand.
   */
  constexpr MultExpression(ExpressionType type, ExpressionPtr lhs,
                           ExpressionPtr rhs)
      : Expression{type, std::move(lhs), std::move(rhs)} {
    value = MultExpression::Value(args[0]->value, args[1]->value);
  }

  double Value(double lhs, double rhs) const override { return lhs * rhs; }

  double GradientValueLhs([[maybe_unused]] double lhs, double rhs,
                          double parentAdjoint) const override {
    return parentAdjoint * rhs;
  }

  double GradientValueRhs(double lhs, [[maybe_unused]] double rhs,
                          double parentAdjoint) const override {
    return parentAdjoint * lhs;
  }

  ExpressionPtr GradientLhs([[maybe_unused]] const ExpressionPtr& lhs,
                            const ExpressionPtr& rhs,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * rhs;
  }

  ExpressionPtr GradientRhs(const ExpressionPtr& lhs,
                            [[maybe_unused]] const ExpressionPtr& rhs,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * lhs;
  }
};

struct UnaryMinusExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param type The expression's type.
   * @param lhs Unary operator's operand.
   */
  constexpr UnaryMinusExpression(ExpressionType type, ExpressionPtr lhs)
      : Expression{type, std::move(lhs)} {
    value = UnaryMinusExpression::Value(args[0]->value, 0.0);
  }

  double Value(double lhs, double) const override { return -lhs; }

  double GradientValueLhs(double, double, double parentAdjoint) const override {
    return -parentAdjoint;
  }

  ExpressionPtr GradientLhs(const ExpressionPtr&, const ExpressionPtr&,
                            const ExpressionPtr& parentAdjoint) const override {
    return -parentAdjoint;
  }
};

inline ExpressionPtr exp(const ExpressionPtr& x);
inline ExpressionPtr sin(const ExpressionPtr& x);
inline ExpressionPtr sinh(const ExpressionPtr& x);
inline ExpressionPtr sqrt(const ExpressionPtr& x);

/**
 * Refcount increment for intrusive shared pointer.
 *
 * @param expr The shared pointer's managed object.
 */
inline constexpr void IntrusiveSharedPtrIncRefCount(Expression* expr) {
  ++expr->refCount;
}

/**
 * Refcount decrement for intrusive shared pointer.
 *
 * @param expr The shared pointer's managed object.
 */
inline constexpr void IntrusiveSharedPtrDecRefCount(Expression* expr) {
  // If a deeply nested tree is being deallocated all at once, calling the
  // Expression destructor when expr's refcount reaches zero can cause a stack
  // overflow. Instead, we iterate over its children to decrement their
  // refcounts and deallocate them.
  small_vector<Expression*> stack;
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
      if constexpr (kUsePoolAllocator) {
        auto alloc = GlobalPoolAllocator<Expression>();
        std::allocator_traits<decltype(alloc)>::deallocate(alloc, elem,
                                                           sizeof(Expression));
      }
    }
  }
}

struct AbsExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param type The expression's type.
   * @param lhs Unary operator's operand.
   */
  constexpr AbsExpression(ExpressionType type, ExpressionPtr lhs)
      : Expression{type, std::move(lhs)} {
    value = AbsExpression::Value(args[0]->value, 0.0);
  }

  double Value(double x, double) const override { return std::abs(x); }

  double GradientValueLhs(double x, double,
                          double parentAdjoint) const override {
    if (x < 0.0) {
      return -parentAdjoint;
    } else if (x > 0.0) {
      return parentAdjoint;
    } else {
      return 0.0;
    }
  }

  ExpressionPtr GradientLhs(const ExpressionPtr& x, const ExpressionPtr&,
                            const ExpressionPtr& parentAdjoint) const override {
    if (x->value < 0.0) {
      return -parentAdjoint;
    } else if (x->value > 0.0) {
      return parentAdjoint;
    } else {
      // Return zero
      return MakeExpressionPtr<ConstExpression>();
    }
  }
};

/**
 * std::abs() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr abs(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    // Return zero
    return x;
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::abs(x->value));
  }

  return MakeExpressionPtr<AbsExpression>(kNonlinear, x);
}

struct AcosExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param type The expression's type.
   * @param lhs Unary operator's operand.
   */
  constexpr AcosExpression(ExpressionType type, ExpressionPtr lhs)
      : Expression{type, std::move(lhs)} {
    value = AcosExpression::Value(args[0]->value, 0.0);
  }

  double Value(double x, double) const override { return std::acos(x); }

  double GradientValueLhs(double x, double,
                          double parentAdjoint) const override {
    return -parentAdjoint / std::sqrt(1.0 - x * x);
  }

  ExpressionPtr GradientLhs(const ExpressionPtr& x, const ExpressionPtr&,
                            const ExpressionPtr& parentAdjoint) const override {
    return -parentAdjoint /
           sleipnir::detail::sqrt(MakeExpressionPtr<ConstExpression>(1.0) -
                                  x * x);
  }
};

/**
 * std::acos() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr acos(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return MakeExpressionPtr<ConstExpression>(std::numbers::pi / 2.0);
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::acos(x->value));
  }

  return MakeExpressionPtr<AcosExpression>(kNonlinear, x);
}

struct AsinExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param type The expression's type.
   * @param lhs Unary operator's operand.
   */
  constexpr AsinExpression(ExpressionType type, ExpressionPtr lhs)
      : Expression{type, std::move(lhs)} {
    value = AsinExpression::Value(args[0]->value, 0.0);
  }

  double Value(double x, double) const override { return std::asin(x); }

  double GradientValueLhs(double x, double,
                          double parentAdjoint) const override {
    return parentAdjoint / std::sqrt(1.0 - x * x);
  }

  ExpressionPtr GradientLhs(const ExpressionPtr& x, const ExpressionPtr&,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint / sleipnir::detail::sqrt(
                               MakeExpressionPtr<ConstExpression>(1.0) - x * x);
  }
};

/**
 * std::asin() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr asin(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    // Return zero
    return x;
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::asin(x->value));
  }

  return MakeExpressionPtr<AsinExpression>(kNonlinear, x);
}

struct AtanExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param type The expression's type.
   * @param lhs Unary operator's operand.
   */
  constexpr AtanExpression(ExpressionType type, ExpressionPtr lhs)
      : Expression{type, std::move(lhs)} {
    value = AtanExpression::Value(args[0]->value, 0.0);
  }

  double Value(double x, double) const override { return std::atan(x); }

  double GradientValueLhs(double x, double,
                          double parentAdjoint) const override {
    return parentAdjoint / (1.0 + x * x);
  }

  ExpressionPtr GradientLhs(const ExpressionPtr& x, const ExpressionPtr&,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint / (MakeExpressionPtr<ConstExpression>(1.0) + x * x);
  }
};

/**
 * std::atan() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr atan(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    // Return zero
    return x;
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::atan(x->value));
  }

  return MakeExpressionPtr<AtanExpression>(kNonlinear, x);
}

struct Atan2Expression final : Expression {
  /**
   * Constructs a binary expression (an operator with two arguments).
   *
   * @param type The expression's type.
   * @param lhs Binary operator's left operand.
   * @param rhs Binary operator's right operand.
   */
  constexpr Atan2Expression(ExpressionType type, ExpressionPtr lhs,
                            ExpressionPtr rhs)
      : Expression{type, std::move(lhs), std::move(rhs)} {
    value = Atan2Expression::Value(args[0]->value, args[1]->value);
  }

  double Value(double y, double x) const override { return std::atan2(y, x); }

  double GradientValueLhs(double y, double x,
                          double parentAdjoint) const override {
    return parentAdjoint * x / (y * y + x * x);
  }

  double GradientValueRhs(double y, double x,
                          double parentAdjoint) const override {
    return parentAdjoint * -y / (y * y + x * x);
  }

  ExpressionPtr GradientLhs(const ExpressionPtr& y, const ExpressionPtr& x,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * x / (y * y + x * x);
  }

  ExpressionPtr GradientRhs(const ExpressionPtr& y, const ExpressionPtr& x,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * -y / (y * y + x * x);
  }
};

/**
 * std::atan2() for Expressions.
 *
 * @param y The y argument.
 * @param x The x argument.
 */
inline ExpressionPtr atan2(const ExpressionPtr& y, const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (y->IsConstant(0.0)) {
    // Return zero
    return y;
  } else if (x->IsConstant(0.0)) {
    return MakeExpressionPtr<ConstExpression>(std::numbers::pi / 2.0);
  }

  // Evaluate constant
  if (y->type == kConstant && x->type == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::atan2(y->value, x->value));
  }

  return MakeExpressionPtr<Atan2Expression>(kNonlinear, y, x);
}

struct CosExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param type The expression's type.
   * @param lhs Unary operator's operand.
   */
  constexpr CosExpression(ExpressionType type, ExpressionPtr lhs)
      : Expression{type, std::move(lhs)} {
    value = CosExpression::Value(args[0]->value, 0.0);
  }

  double Value(double x, double) const override { return std::cos(x); }

  double GradientValueLhs(double x, double,
                          double parentAdjoint) const override {
    return -parentAdjoint * std::sin(x);
  }

  ExpressionPtr GradientLhs(const ExpressionPtr& x, const ExpressionPtr&,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * -sleipnir::detail::sin(x);
  }
};

/**
 * std::cos() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr cos(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return MakeExpressionPtr<ConstExpression>(1.0);
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::cos(x->value));
  }

  return MakeExpressionPtr<CosExpression>(kNonlinear, x);
}

struct CoshExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param type The expression's type.
   * @param lhs Unary operator's operand.
   */
  constexpr CoshExpression(ExpressionType type, ExpressionPtr lhs)
      : Expression{type, std::move(lhs)} {
    value = CoshExpression::Value(args[0]->value, 0.0);
  }

  double Value(double x, double) const override { return std::cosh(x); }

  double GradientValueLhs(double x, double,
                          double parentAdjoint) const override {
    return parentAdjoint * std::sinh(x);
  }

  ExpressionPtr GradientLhs(const ExpressionPtr& x, const ExpressionPtr&,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * sleipnir::detail::sinh(x);
  }
};

/**
 * std::cosh() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr cosh(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return MakeExpressionPtr<ConstExpression>(1.0);
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::cosh(x->value));
  }

  return MakeExpressionPtr<CoshExpression>(kNonlinear, x);
}

struct ErfExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param type The expression's type.
   * @param lhs Unary operator's operand.
   */
  constexpr ErfExpression(ExpressionType type, ExpressionPtr lhs)
      : Expression{type, std::move(lhs)} {
    value = ErfExpression::Value(args[0]->value, 0.0);
  }

  double Value(double x, double) const override { return std::erf(x); }

  double GradientValueLhs(double x, double,
                          double parentAdjoint) const override {
    return parentAdjoint * 2.0 * std::numbers::inv_sqrtpi * std::exp(-x * x);
  }

  ExpressionPtr GradientLhs(const ExpressionPtr& x, const ExpressionPtr&,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint *
           MakeExpressionPtr<ConstExpression>(2.0 * std::numbers::inv_sqrtpi) *
           sleipnir::detail::exp(-x * x);
  }
};

/**
 * std::erf() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr erf(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    // Return zero
    return x;
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::erf(x->value));
  }

  return MakeExpressionPtr<ErfExpression>(kNonlinear, x);
}

struct ExpExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param type The expression's type.
   * @param lhs Unary operator's operand.
   */
  constexpr ExpExpression(ExpressionType type, ExpressionPtr lhs)
      : Expression{type, std::move(lhs)} {
    value = ExpExpression::Value(args[0]->value, 0.0);
  }

  double Value(double x, double) const override { return std::exp(x); }

  double GradientValueLhs(double x, double,
                          double parentAdjoint) const override {
    return parentAdjoint * std::exp(x);
  }

  ExpressionPtr GradientLhs(const ExpressionPtr& x, const ExpressionPtr&,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * sleipnir::detail::exp(x);
  }
};

/**
 * std::exp() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr exp(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return MakeExpressionPtr<ConstExpression>(1.0);
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::exp(x->value));
  }

  return MakeExpressionPtr<ExpExpression>(kNonlinear, x);
}

inline ExpressionPtr hypot(const ExpressionPtr& x, const ExpressionPtr& y);

struct HypotExpression final : Expression {
  /**
   * Constructs a binary expression (an operator with two arguments).
   *
   * @param type The expression's type.
   * @param lhs Binary operator's left operand.
   * @param rhs Binary operator's right operand.
   */
  constexpr HypotExpression(ExpressionType type, ExpressionPtr lhs,
                            ExpressionPtr rhs)
      : Expression{type, std::move(lhs), std::move(rhs)} {
    value = HypotExpression::Value(args[0]->value, args[1]->value);
  }

  double Value(double x, double y) const override { return std::hypot(x, y); }

  double GradientValueLhs(double x, double y,
                          double parentAdjoint) const override {
    return parentAdjoint * x / std::hypot(x, y);
  }

  double GradientValueRhs(double x, double y,
                          double parentAdjoint) const override {
    return parentAdjoint * y / std::hypot(x, y);
  }

  ExpressionPtr GradientLhs(const ExpressionPtr& x, const ExpressionPtr& y,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * x / sleipnir::detail::hypot(x, y);
  }

  ExpressionPtr GradientRhs(const ExpressionPtr& x, const ExpressionPtr& y,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * y / sleipnir::detail::hypot(x, y);
  }
};

/**
 * std::hypot() for Expressions.
 *
 * @param x The x argument.
 * @param y The y argument.
 */
inline ExpressionPtr hypot(const ExpressionPtr& x, const ExpressionPtr& y) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    return y;
  } else if (y->IsConstant(0.0)) {
    return x;
  }

  // Evaluate constant
  if (x->type == kConstant && y->type == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::hypot(x->value, y->value));
  }

  return MakeExpressionPtr<HypotExpression>(kNonlinear, x, y);
}

struct LogExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param type The expression's type.
   * @param lhs Unary operator's operand.
   */
  constexpr LogExpression(ExpressionType type, ExpressionPtr lhs)
      : Expression{type, std::move(lhs)} {
    value = LogExpression::Value(args[0]->value, 0.0);
  }

  double Value(double x, double) const override { return std::log(x); }

  double GradientValueLhs(double x, double,
                          double parentAdjoint) const override {
    return parentAdjoint / x;
  }

  ExpressionPtr GradientLhs(const ExpressionPtr& x, const ExpressionPtr&,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint / x;
  }
};

/**
 * std::log() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr log(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    // Return zero
    return x;
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::log(x->value));
  }

  return MakeExpressionPtr<LogExpression>(kNonlinear, x);
}

struct Log10Expression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param type The expression's type.
   * @param lhs Unary operator's operand.
   */
  constexpr Log10Expression(ExpressionType type, ExpressionPtr lhs)
      : Expression{type, std::move(lhs)} {
    value = Log10Expression::Value(args[0]->value, 0.0);
  }

  double Value(double x, double) const override { return std::log10(x); }

  double GradientValueLhs(double x, double,
                          double parentAdjoint) const override {
    return parentAdjoint / (std::numbers::ln10 * x);
  }

  ExpressionPtr GradientLhs(const ExpressionPtr& x, const ExpressionPtr&,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint /
           (MakeExpressionPtr<ConstExpression>(std::numbers::ln10) * x);
  }
};

/**
 * std::log10() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr log10(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    // Return zero
    return x;
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::log10(x->value));
  }

  return MakeExpressionPtr<Log10Expression>(kNonlinear, x);
}

inline ExpressionPtr pow(const ExpressionPtr& base, const ExpressionPtr& power);

struct PowExpression final : Expression {
  /**
   * Constructs a binary expression (an operator with two arguments).
   *
   * @param type The expression's type.
   * @param lhs Binary operator's left operand.
   * @param rhs Binary operator's right operand.
   */
  constexpr PowExpression(ExpressionType type, ExpressionPtr lhs,
                          ExpressionPtr rhs)
      : Expression{type, std::move(lhs), std::move(rhs)} {
    value = PowExpression::Value(args[0]->value, args[1]->value);
  }

  double Value(double base, double power) const override {
    return std::pow(base, power);
  }

  double GradientValueLhs(double base, double power,
                          double parentAdjoint) const override {
    return parentAdjoint * std::pow(base, power - 1) * power;
  }

  double GradientValueRhs(double base, double power,
                          double parentAdjoint) const override {
    // Since x * std::log(x) -> 0 as x -> 0
    if (base == 0.0) {
      return 0.0;
    } else {
      return parentAdjoint * std::pow(base, power - 1) * base * std::log(base);
    }
  }

  ExpressionPtr GradientLhs(const ExpressionPtr& base,
                            const ExpressionPtr& power,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint *
           sleipnir::detail::pow(
               base, power - MakeExpressionPtr<ConstExpression>(1.0)) *
           power;
  }

  ExpressionPtr GradientRhs(const ExpressionPtr& base,
                            const ExpressionPtr& power,
                            const ExpressionPtr& parentAdjoint) const override {
    // Since x * std::log(x) -> 0 as x -> 0
    if (base->value == 0.0) {
      // Return zero
      return base;
    } else {
      return parentAdjoint *
             sleipnir::detail::pow(
                 base, power - MakeExpressionPtr<ConstExpression>(1.0)) *
             base * sleipnir::detail::log(base);
    }
  }
};

/**
 * std::pow() for Expressions.
 *
 * @param base The base.
 * @param power The power.
 */
inline ExpressionPtr pow(const ExpressionPtr& base,
                         const ExpressionPtr& power) {
  using enum ExpressionType;

  // Prune expression
  if (base->IsConstant(0.0)) {
    // Return zero
    return base;
  } else if (base->IsConstant(1.0)) {
    return base;
  }
  if (power->IsConstant(0.0)) {
    return MakeExpressionPtr<ConstExpression>(1.0);
  } else if (power->IsConstant(1.0)) {
    return base;
  }

  // Evaluate constant
  if (base->type == kConstant && power->type == kConstant) {
    return MakeExpressionPtr<ConstExpression>(
        std::pow(base->value, power->value));
  }

  return MakeExpressionPtr<PowExpression>(
      base->type == kLinear && power->IsConstant(2.0) ? kQuadratic : kNonlinear,
      base, power);
}

struct SignExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param type The expression's type.
   * @param lhs Unary operator's operand.
   */
  constexpr SignExpression(ExpressionType type, ExpressionPtr lhs)
      : Expression{type, std::move(lhs)} {
    value = SignExpression::Value(args[0]->value, 0.0);
  }

  double Value(double x, double) const override {
    if (x < 0.0) {
      return -1.0;
    } else if (x == 0.0) {
      return 0.0;
    } else {
      return 1.0;
    }
  }

  double GradientValueLhs(double, double, double) const override { return 0.0; }

  ExpressionPtr GradientLhs(const ExpressionPtr&, const ExpressionPtr&,
                            const ExpressionPtr&) const override {
    // Return zero
    return MakeExpressionPtr<ConstExpression>();
  }
};

/**
 * sign() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr sign(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Evaluate constant
  if (x->type == kConstant) {
    if (x->value < 0.0) {
      return MakeExpressionPtr<ConstExpression>(-1.0);
    } else if (x->value == 0.0) {
      // Return zero
      return x;
    } else {
      return MakeExpressionPtr<ConstExpression>(1.0);
    }
  }

  return MakeExpressionPtr<SignExpression>(kNonlinear, x);
}

struct SinExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param type The expression's type.
   * @param lhs Unary operator's operand.
   */
  constexpr SinExpression(ExpressionType type, ExpressionPtr lhs)
      : Expression{type, std::move(lhs)} {
    value = SinExpression::Value(args[0]->value, 0.0);
  }

  double Value(double x, double) const override { return std::sin(x); }

  double GradientValueLhs(double x, double,
                          double parentAdjoint) const override {
    return parentAdjoint * std::cos(x);
  }

  ExpressionPtr GradientLhs(const ExpressionPtr& x, const ExpressionPtr&,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * sleipnir::detail::cos(x);
  }
};

/**
 * std::sin() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr sin(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    // Return zero
    return x;
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::sin(x->value));
  }

  return MakeExpressionPtr<SinExpression>(kNonlinear, x);
}

struct SinhExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param type The expression's type.
   * @param lhs Unary operator's operand.
   */
  constexpr SinhExpression(ExpressionType type, ExpressionPtr lhs)
      : Expression{type, std::move(lhs)} {
    value = SinhExpression::Value(args[0]->value, 0.0);
  }

  double Value(double x, double) const override { return std::sinh(x); }

  double GradientValueLhs(double x, double,
                          double parentAdjoint) const override {
    return parentAdjoint * std::cosh(x);
  }

  ExpressionPtr GradientLhs(const ExpressionPtr& x, const ExpressionPtr&,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint * sleipnir::detail::cosh(x);
  }
};

/**
 * std::sinh() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr sinh(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    // Return zero
    return x;
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::sinh(x->value));
  }

  return MakeExpressionPtr<SinhExpression>(kNonlinear, x);
}

struct SqrtExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param type The expression's type.
   * @param lhs Unary operator's operand.
   */
  constexpr SqrtExpression(ExpressionType type, ExpressionPtr lhs)
      : Expression{type, std::move(lhs)} {
    value = SqrtExpression::Value(args[0]->value, 0.0);
  }

  double Value(double x, double) const override { return std::sqrt(x); }

  double GradientValueLhs(double x, double,
                          double parentAdjoint) const override {
    return parentAdjoint / (2.0 * std::sqrt(x));
  }

  ExpressionPtr GradientLhs(const ExpressionPtr& x, const ExpressionPtr&,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint / (MakeExpressionPtr<ConstExpression>(2.0) *
                            sleipnir::detail::sqrt(x));
  }
};

/**
 * std::sqrt() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr sqrt(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Evaluate constant
  if (x->type == kConstant) {
    if (x->value == 0.0) {
      // Return zero
      return x;
    } else if (x->value == 1.0) {
      return x;
    } else {
      return MakeExpressionPtr<ConstExpression>(std::sqrt(x->value));
    }
  }

  return MakeExpressionPtr<SqrtExpression>(kNonlinear, x);
}

struct TanExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param type The expression's type.
   * @param lhs Unary operator's operand.
   */
  constexpr TanExpression(ExpressionType type, ExpressionPtr lhs)
      : Expression{type, std::move(lhs)} {
    value = TanExpression::Value(args[0]->value, 0.0);
  }

  double Value(double x, double) const override { return std::tan(x); }

  double GradientValueLhs(double x, double,
                          double parentAdjoint) const override {
    return parentAdjoint / (std::cos(x) * std::cos(x));
  }

  ExpressionPtr GradientLhs(const ExpressionPtr& x, const ExpressionPtr&,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint /
           (sleipnir::detail::cos(x) * sleipnir::detail::cos(x));
  }
};

/**
 * std::tan() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr tan(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    // Return zero
    return x;
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::tan(x->value));
  }

  return MakeExpressionPtr<TanExpression>(kNonlinear, x);
}

struct TanhExpression final : Expression {
  /**
   * Constructs an unary expression (an operator with one argument).
   *
   * @param type The expression's type.
   * @param lhs Unary operator's operand.
   */
  constexpr TanhExpression(ExpressionType type, ExpressionPtr lhs)
      : Expression{type, std::move(lhs)} {
    value = TanhExpression::Value(args[0]->value, 0.0);
  }

  double Value(double x, double) const override { return std::tanh(x); }

  double GradientValueLhs(double x, double,
                          double parentAdjoint) const override {
    return parentAdjoint / (std::cosh(x) * std::cosh(x));
  }

  ExpressionPtr GradientLhs(const ExpressionPtr& x, const ExpressionPtr&,
                            const ExpressionPtr& parentAdjoint) const override {
    return parentAdjoint /
           (sleipnir::detail::cosh(x) * sleipnir::detail::cosh(x));
  }
};

/**
 * std::tanh() for Expressions.
 *
 * @param x The argument.
 */
inline ExpressionPtr tanh(const ExpressionPtr& x) {
  using enum ExpressionType;

  // Prune expression
  if (x->IsConstant(0.0)) {
    // Return zero
    return x;
  }

  // Evaluate constant
  if (x->type == kConstant) {
    return MakeExpressionPtr<ConstExpression>(std::tanh(x->value));
  }

  return MakeExpressionPtr<TanhExpression>(kNonlinear, x);
}

}  // namespace sleipnir::detail
