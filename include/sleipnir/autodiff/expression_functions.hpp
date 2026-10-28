// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <string_view>
#include <type_traits>
#include <utility>

#include "sleipnir/autodiff/expression.hpp"
#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/util/function_ref.hpp"

namespace slp::detail {

/// Derived expression type for abs().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct AbsExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr AbsExpression(ExpressionPtr<Scalar> x) : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::abs;
    return abs(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "abs"; }

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

 private:
  Scalar grad_l() const {
    if (x->val < Scalar(0)) {
      return -this->adj;
    } else if (x->val > Scalar(0)) {
      return this->adj;
    } else {
      return Scalar(0);
    }
  }

  ExpressionPtr<Scalar> grad_expr_l() const { return this->adj_expr * sign(x); }
};

/// abs() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> abs(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::abs;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    // Return zero, which x currently is
    return x;
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(abs(x->val));
  }

  return make_expression_ptr<AbsExpression<Scalar>>(x);
}

/// Derived expression type for acos().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct AcosExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr AcosExpression(ExpressionPtr<Scalar> x)
      : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::acos;
    return acos(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "acos"; }

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

 private:
  Scalar grad_l() const {
    using std::sqrt;
    return -this->adj / sqrt(Scalar(1) - x->val * x->val);
  }

  ExpressionPtr<Scalar> grad_expr_l() const {
    return -this->adj_expr / sqrt(constant_ptr(Scalar(1)) - x * x);
  }
};

/// acos() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> acos(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::acos;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    return constant_ptr(Scalar(std::numbers::pi) / Scalar(2));
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(acos(x->val));
  }

  return make_expression_ptr<AcosExpression<Scalar>>(x);
}

/// Derived expression type for asin().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct AsinExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr AsinExpression(ExpressionPtr<Scalar> x)
      : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::asin;
    return asin(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "asin"; }

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

 private:
  Scalar grad_l() const {
    using std::sqrt;
    return this->adj / sqrt(Scalar(1) - x->val * x->val);
  }

  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adj_expr / sqrt(constant_ptr(Scalar(1)) - x * x);
  }
};

/// asin() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> asin(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::asin;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    // Return zero, which x currently is
    return x;
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(asin(x->val));
  }

  return make_expression_ptr<AsinExpression<Scalar>>(x);
}

/// Derived expression type for atan().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct AtanExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr AtanExpression(ExpressionPtr<Scalar> x)
      : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::atan;
    return atan(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "atan"; }

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

 private:
  Scalar grad_l() const { return this->adj / (Scalar(1) + x->val * x->val); }

  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adj_expr / (constant_ptr(Scalar(1)) + x * x);
  }
};

/// atan() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> atan(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::atan;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    // Return zero, which x currently is
    return x;
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(atan(x->val));
  }

  return make_expression_ptr<AtanExpression<Scalar>>(x);
}

/// Derived expression type for atan2().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct Atan2Expression final : Expression<Scalar> {
  /// Binary operator's left operand.
  ExpressionPtr<Scalar> y;

  /// Binary operator's right operand.
  ExpressionPtr<Scalar> x;

  /// Constructs a binary expression (an operator with two arguments).
  ///
  /// @param y Binary operator's left operand.
  /// @param x Binary operator's right operand.
  constexpr Atan2Expression(ExpressionPtr<Scalar> y, ExpressionPtr<Scalar> x)
      : y{std::move(y)}, x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(y.get());
    func(x.get());
  }

  Scalar value() const override {
    using std::atan2;
    return atan2(y->val, x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "atan2"; }

  void accumulate_adjoints() const override {
    y->adj += grad_l();
    x->adj += grad_r();
  }

  void accumulate_adjoints_expr() const override {
    y->adj_expr += grad_expr_l();
    x->adj_expr += grad_expr_r();
  }

 private:
  Scalar grad_l() const {
    return this->adj * x->val / (y->val * y->val + x->val * x->val);
  }

  Scalar grad_r() const {
    return this->adj * -y->val / (y->val * y->val + x->val * x->val);
  }

  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adj_expr * x / (y * y + x * x);
  }

  ExpressionPtr<Scalar> grad_expr_r() const {
    return this->adj_expr * -y / (y * y + x * x);
  }
};

/// atan2() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param y The y argument.
/// @param x The x argument.
template <typename Scalar>
ExpressionPtr<Scalar> atan2(const ExpressionPtr<Scalar>& y,
                            const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::atan2;

  // Evaluate constant
  if (y->type() == CONSTANT && x->type() == CONSTANT) {
    return constant_ptr(atan2(y->val, x->val));
  }

  return make_expression_ptr<Atan2Expression<Scalar>>(y, x);
}

/// Derived expression type for cbrt().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct CbrtExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr CbrtExpression(ExpressionPtr<Scalar> x)
      : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::cbrt;
    return cbrt(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "cbrt"; }

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

 private:
  Scalar grad_l() const {
    using std::cbrt;

    Scalar c = cbrt(x->val);
    return this->adj / (Scalar(3) * c * c);
  }

  ExpressionPtr<Scalar> grad_expr_l() const {
    auto c = cbrt(x);
    return this->adj_expr / (constant_ptr(Scalar(3)) * c * c);
  }
};

/// cbrt() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> cbrt(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::cbrt;

  // Evaluate constant
  if (x->type() == CONSTANT) {
    if (x->val == Scalar(0)) {
      // Return zero
      return x;
    } else if (x->val == Scalar(-1) || x->val == Scalar(1)) {
      return x;
    } else {
      return constant_ptr(cbrt(x->val));
    }
  }

  return make_expression_ptr<CbrtExpression<Scalar>>(x);
}

/// Derived expression type for cos().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct CosExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr CosExpression(ExpressionPtr<Scalar> x) : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::cos;
    return cos(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "cos"; }

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

 private:
  Scalar grad_l() const {
    using std::sin;
    return this->adj * -sin(x->val);
  }

  ExpressionPtr<Scalar> grad_expr_l() const { return this->adj_expr * -sin(x); }
};

/// cos() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> cos(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::cos;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    return constant_ptr(Scalar(1));
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(cos(x->val));
  }

  return make_expression_ptr<CosExpression<Scalar>>(x);
}

/// Derived expression type for cosh().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct CoshExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr CoshExpression(ExpressionPtr<Scalar> x)
      : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::cosh;
    return cosh(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "cosh"; }

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

 private:
  Scalar grad_l() const {
    using std::sinh;
    return this->adj * sinh(x->val);
  }

  ExpressionPtr<Scalar> grad_expr_l() const { return this->adj_expr * sinh(x); }
};

/// cosh() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> cosh(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::cosh;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    return constant_ptr(Scalar(1));
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(cosh(x->val));
  }

  return make_expression_ptr<CoshExpression<Scalar>>(x);
}

/// Derived expression type for erf().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct ErfExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr ErfExpression(ExpressionPtr<Scalar> x) : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::erf;
    return erf(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "erf"; }

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

 private:
  Scalar grad_l() const {
    using std::exp;
    return this->adj * Scalar(2.0 * std::numbers::inv_sqrtpi) *
           exp(-x->val * x->val);
  }

  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adj_expr *
           constant_ptr(Scalar(2.0 * std::numbers::inv_sqrtpi)) * exp(-x * x);
  }
};

/// erf() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> erf(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::erf;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    // Return zero, which x currently is
    return x;
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(erf(x->val));
  }

  return make_expression_ptr<ErfExpression<Scalar>>(x);
}

/// Derived expression type for exp().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct ExpExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr ExpExpression(ExpressionPtr<Scalar> x) : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::exp;
    return exp(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "exp"; }

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

 private:
  Scalar grad_l() const {
    using std::exp;
    return this->adj * exp(x->val);
  }

  ExpressionPtr<Scalar> grad_expr_l() const { return this->adj_expr * exp(x); }
};

/// exp() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> exp(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::exp;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    return constant_ptr(Scalar(1));
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(exp(x->val));
  }

  return make_expression_ptr<ExpExpression<Scalar>>(x);
}

template <typename Scalar>
ExpressionPtr<Scalar> hypot(const ExpressionPtr<Scalar>& x,
                            const ExpressionPtr<Scalar>& y);

/// Derived expression type for hypot() with two arguments.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct Hypot2Expression final : Expression<Scalar> {
  /// Binary operator's left operand.
  ExpressionPtr<Scalar> x;

  /// Binary operator's right operand.
  ExpressionPtr<Scalar> y;

  /// Constructs a binary expression (an operator with two arguments).
  ///
  /// @param x Binary operator's left operand.
  /// @param y Binary operator's right operand.
  constexpr Hypot2Expression(ExpressionPtr<Scalar> x, ExpressionPtr<Scalar> y)
      : x{std::move(x)}, y{std::move(y)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
    func(y.get());
  }

  Scalar value() const override {
    using std::hypot;
    return hypot(x->val, y->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "hypot"; }

  void accumulate_adjoints() const override {
    using std::hypot;
    Scalar norm = hypot(x->val, y->val);
    x->adj += this->adj * x->val / norm;
    y->adj += this->adj * y->val / norm;
  }

  void accumulate_adjoints_expr() const override {
    auto norm = hypot(x, y);
    x->adj_expr += this->adj_expr * x / norm;
    y->adj_expr += this->adj_expr * y / norm;
  }
};

/// hypot() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The x argument.
/// @param y The y argument.
template <typename Scalar>
ExpressionPtr<Scalar> hypot(const ExpressionPtr<Scalar>& x,
                            const ExpressionPtr<Scalar>& y) {
  using enum ExpressionType;
  using std::hypot;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    return abs(y);
  } else if (y->is_constant(Scalar(0))) {
    return abs(x);
  }

  // Evaluate constant
  if (x->type() == CONSTANT && y->type() == CONSTANT) {
    return constant_ptr(hypot(x->val, y->val));
  }

  return make_expression_ptr<Hypot2Expression<Scalar>>(x, y);
}

template <typename Scalar>
ExpressionPtr<Scalar> hypot(const ExpressionPtr<Scalar>& x,
                            const ExpressionPtr<Scalar>& y,
                            const ExpressionPtr<Scalar>& z);

/// Derived expression type for hypot() with three arguments.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct Hypot3Expression final : Expression<Scalar> {
  /// Ternary operator's first operand.
  ExpressionPtr<Scalar> x;

  /// Ternary operator's second operand.
  ExpressionPtr<Scalar> y;

  /// Ternary operator's third operand.
  ExpressionPtr<Scalar> z;

  /// Constructs a ternary expression (an operator with three arguments).
  ///
  /// @param x Ternary operator's first operand.
  /// @param y Ternary operator's second operand.
  /// @param z Ternary operator's third operand.
  constexpr Hypot3Expression(ExpressionPtr<Scalar> x, ExpressionPtr<Scalar> y,
                             ExpressionPtr<Scalar> z)
      : x{std::move(x)}, y{std::move(y)}, z{std::move(z)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
    func(y.get());
    func(z.get());
  }

  Scalar value() const override {
    using std::hypot;
    return hypot(x->val, y->val, z->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "hypot"; }

  void accumulate_adjoints() const override {
    using std::hypot;
    Scalar norm = hypot(x->val, y->val, z->val);
    x->adj += this->adj * x->val / norm;
    y->adj += this->adj * y->val / norm;
    z->adj += this->adj * z->val / norm;
  }

  void accumulate_adjoints_expr() const override {
    auto norm = hypot(x, y, z);
    x->adj_expr += this->adj_expr * x / norm;
    y->adj_expr += this->adj_expr * y / norm;
    z->adj_expr += this->adj_expr * z / norm;
  }
};

/// hypot() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The x argument.
/// @param y The y argument.
/// @param z The z argument.
template <typename Scalar>
ExpressionPtr<Scalar> hypot(const ExpressionPtr<Scalar>& x,
                            const ExpressionPtr<Scalar>& y,
                            const ExpressionPtr<Scalar>& z) {
  using enum ExpressionType;
  using std::hypot;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    return hypot(y, z);
  } else if (y->is_constant(Scalar(0))) {
    return hypot(x, z);
  } else if (z->is_constant(Scalar(0))) {
    return hypot(x, y);
  }

  // Evaluate constant
  if (x->type() == CONSTANT && y->type() == CONSTANT && z->type() == CONSTANT) {
    return constant_ptr(hypot(x->val, y->val, z->val));
  }

  return make_expression_ptr<Hypot3Expression<Scalar>>(x, y, z);
}

/// Derived expression type for if_else().
///
/// Returns t if cond(a, b) is true, otherwise f.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct IfElseExpression final : Expression<Scalar> {
  /// Condition evaluated on a and b.
  bool (*cond)(Scalar a, Scalar b);

  /// The condition's first argument.
  ExpressionPtr<Scalar> a;

  /// The condition's second argument.
  ExpressionPtr<Scalar> b;

  /// Value selected when cond(a, b) is true.
  ExpressionPtr<Scalar> t;

  /// Value selected when cond(a, b) is false.
  ExpressionPtr<Scalar> f;

  /// Constructs an if-else expression.
  ///
  /// @param cond Condition evaluated on a and b.
  /// @param a The condition's first argument.
  /// @param b The condition's second argument.
  /// @param t Value selected when cond(a, b) is true.
  /// @param f Value selected when cond(a, b) is false.
  constexpr IfElseExpression(bool (*cond)(Scalar a, Scalar b),
                             ExpressionPtr<Scalar> a, ExpressionPtr<Scalar> b,
                             ExpressionPtr<Scalar> t, ExpressionPtr<Scalar> f)
      : cond{cond},
        a{std::move(a)},
        b{std::move(b)},
        t{std::move(t)},
        f{std::move(f)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(a.get());
    func(b.get());
    func(t.get());
    func(f.get());
  }

  Scalar value() const override {
    return cond(a->val, b->val) ? t->val : f->val;
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "if-else"; }

  // a and b have zero gradient, so only t and f accumulate adjoints
  void accumulate_adjoints() const override {
    if (cond(a->val, b->val)) {
      t->adj += this->adj;
    } else {
      f->adj += this->adj;
    }
  }

  void accumulate_adjoints_expr() const override {
    t->adj_expr += grad_expr_t();
    f->adj_expr += grad_expr_f();
  }

 private:
  ExpressionPtr<Scalar> grad_expr_t() const {
    // adjoint if cond(a, b), otherwise 0
    return if_else(cond, a, b, this->adj_expr, constant_ptr(Scalar(0)));
  }

  ExpressionPtr<Scalar> grad_expr_f() const {
    // 0 if cond(a, b), otherwise adjoint
    return if_else(cond, a, b, constant_ptr(Scalar(0)), this->adj_expr);
  }
};

/// Returns t if cond(a, b) is true, otherwise f.
///
/// @tparam Scalar Scalar type.
/// @param cond Condition evaluated on a and b.
/// @param a The condition's first argument.
/// @param b The condition's second argument.
/// @param t Value selected when cond(a, b) is true.
/// @param f Value selected when cond(a, b) is false.
template <typename Scalar>
ExpressionPtr<Scalar> if_else(
    std::type_identity_t<bool (*)(Scalar a, Scalar b)> cond,
    const ExpressionPtr<Scalar>& a, const ExpressionPtr<Scalar>& b,
    const ExpressionPtr<Scalar>& t, const ExpressionPtr<Scalar>& f) {
  using enum ExpressionType;

  // Prune expression
  if (t == f) {
    // Return t, which both branches currently are
    return t;
  } else if (t->type() == CONSTANT && f->type() == CONSTANT &&
             t->val == f->val) {
    // Return t, which both branches currently equal
    return t;
  }

  // Evaluate constant condition
  if (a->type() == CONSTANT && b->type() == CONSTANT) {
    return cond(a->val, b->val) ? t : f;
  }

  return make_expression_ptr<IfElseExpression<Scalar>>(cond, a, b, t, f);
}

/// Derived expression type for log().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct LogExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr LogExpression(ExpressionPtr<Scalar> x) : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::log;
    return log(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "log"; }

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

 private:
  Scalar grad_l() const { return this->adj / x->val; }

  ExpressionPtr<Scalar> grad_expr_l() const { return this->adj_expr / x; }
};

/// log() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> log(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::log;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    return constant_ptr(-std::numeric_limits<Scalar>::infinity());
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(log(x->val));
  }

  return make_expression_ptr<LogExpression<Scalar>>(x);
}

/// Derived expression type for log10().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct Log10Expression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr Log10Expression(ExpressionPtr<Scalar> x)
      : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::log10;
    return log10(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "log10"; }

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

 private:
  Scalar grad_l() const {
    return this->adj / (Scalar(std::numbers::ln10) * x->val);
  }

  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adj_expr / (constant_ptr(Scalar(std::numbers::ln10)) * x);
  }
};

/// log10() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> log10(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::log10;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    return constant_ptr(-std::numeric_limits<Scalar>::infinity());
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(log10(x->val));
  }

  return make_expression_ptr<Log10Expression<Scalar>>(x);
}

/// Derived expression type for max().
///
/// Returns the greater of a and b. If the values are equivalent, returns a.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct MaxExpression final : Expression<Scalar> {
  /// Binary operator's left operand.
  ExpressionPtr<Scalar> a;

  /// Binary operator's right operand.
  ExpressionPtr<Scalar> b;

  /// Constructs a binary expression (an operator with two arguments).
  ///
  /// @param a Binary operator's left operand.
  /// @param b Binary operator's right operand.
  constexpr MaxExpression(ExpressionPtr<Scalar> a, ExpressionPtr<Scalar> b)
      : a{std::move(a)}, b{std::move(b)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(a.get());
    func(b.get());
  }

  Scalar value() const override {
    using std::max;
    return max(a->val, b->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "max"; }

  void accumulate_adjoints() const override {
    a->adj += grad_l();
    b->adj += grad_r();
  }

  void accumulate_adjoints_expr() const override {
    a->adj_expr += grad_expr_l();
    b->adj_expr += grad_expr_r();
  }

 private:
  Scalar grad_l() const { return a->val >= b->val ? this->adj : Scalar(0); }

  Scalar grad_r() const { return a->val >= b->val ? Scalar(0) : this->adj; }

  ExpressionPtr<Scalar> grad_expr_l() const {
    // adjoint if a >= b, otherwise 0
    return if_else([](Scalar a, Scalar b) { return a >= b; }, a, b,
                   this->adj_expr, constant_ptr(Scalar(0)));
  }

  ExpressionPtr<Scalar> grad_expr_r() const {
    // 0 if a >= b, otherwise adjoint
    return if_else([](Scalar a, Scalar b) { return a >= b; }, a, b,
                   constant_ptr(Scalar(0)), this->adj_expr);
  }
};

/// max() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param a The a argument.
/// @param b The b argument.
template <typename Scalar>
ExpressionPtr<Scalar> max(const ExpressionPtr<Scalar>& a,
                          const ExpressionPtr<Scalar>& b) {
  using enum ExpressionType;
  using std::max;

  // Evaluate constant
  if (a->type() == CONSTANT && b->type() == CONSTANT) {
    return constant_ptr(max(a->val, b->val));
  }

  return make_expression_ptr<MaxExpression<Scalar>>(a, b);
}

/// Derived expression type for min().
///
/// Returns the lesser of a and b. If the values are equivalent, returns a.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct MinExpression final : Expression<Scalar> {
  /// Binary operator's left operand.
  ExpressionPtr<Scalar> a;

  /// Binary operator's right operand.
  ExpressionPtr<Scalar> b;

  /// Constructs a binary expression (an operator with two arguments).
  ///
  /// @param a Binary operator's left operand.
  /// @param b Binary operator's right operand.
  constexpr MinExpression(ExpressionPtr<Scalar> a, ExpressionPtr<Scalar> b)
      : a{std::move(a)}, b{std::move(b)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(a.get());
    func(b.get());
  }

  Scalar value() const override {
    using std::min;
    return min(a->val, b->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "min"; }

  void accumulate_adjoints() const override {
    a->adj += grad_l();
    b->adj += grad_r();
  }

  void accumulate_adjoints_expr() const override {
    a->adj_expr += grad_expr_l();
    b->adj_expr += grad_expr_r();
  }

 private:
  Scalar grad_l() const { return a->val <= b->val ? this->adj : Scalar(0); }

  Scalar grad_r() const { return a->val <= b->val ? Scalar(0) : this->adj; }

  ExpressionPtr<Scalar> grad_expr_l() const {
    // adjoint if a <= b, otherwise 0
    return if_else([](Scalar a, Scalar b) { return a <= b; }, a, b,
                   this->adj_expr, constant_ptr(Scalar(0)));
  }

  ExpressionPtr<Scalar> grad_expr_r() const {
    // 0 if a <= b, otherwise adjoint
    return if_else([](Scalar a, Scalar b) { return a <= b; }, a, b,
                   constant_ptr(Scalar(0)), this->adj_expr);
  }
};

/// min() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param a The a argument.
/// @param b The b argument.
template <typename Scalar>
ExpressionPtr<Scalar> min(const ExpressionPtr<Scalar>& a,
                          const ExpressionPtr<Scalar>& b) {
  using enum ExpressionType;
  using std::min;

  // Evaluate constant
  if (a->type() == CONSTANT && b->type() == CONSTANT) {
    return constant_ptr(min(a->val, b->val));
  }

  return make_expression_ptr<MinExpression<Scalar>>(a, b);
}

template <typename Scalar>
ExpressionPtr<Scalar> pow(const ExpressionPtr<Scalar>& base,
                          const ExpressionPtr<Scalar>& power);

/// Derived expression type for pow().
///
/// @tparam Scalar Scalar type.
/// @tparam T Expression type.
template <typename Scalar, ExpressionType T>
struct PowExpression final : Expression<Scalar> {
  /// Binary operator's left operand.
  ExpressionPtr<Scalar> base;

  /// Binary operator's right operand.
  ExpressionPtr<Scalar> power;

  /// Constructs a binary expression (an operator with two arguments).
  ///
  /// @param base Binary operator's left operand.
  /// @param power Binary operator's right operand.
  constexpr PowExpression(ExpressionPtr<Scalar> base,
                          ExpressionPtr<Scalar> power)
      : base{std::move(base)}, power{std::move(power)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(base.get());
    func(power.get());
  }

  Scalar value() const override {
    using std::pow;
    return pow(base->val, power->val);
  }

  ExpressionType type() const override { return T; }

  std::string_view name() const override { return "pow"; }

  void accumulate_adjoints() const override {
    base->adj += grad_l();
    power->adj += grad_r();
  }

  void accumulate_adjoints_expr() const override {
    base->adj_expr += grad_expr_l();
    power->adj_expr += grad_expr_r();
  }

 private:
  Scalar grad_l() const {
    using std::pow;
    return this->adj * pow(base->val, power->val - Scalar(1)) * power->val;
  }

  Scalar grad_r() const {
    using std::log;
    using std::pow;

    return this->adj * pow(base->val, power->val) * log(base->val);
  }

  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adj_expr * pow(base, power - constant_ptr(Scalar(1))) * power;
  }

  ExpressionPtr<Scalar> grad_expr_r() const {
    return this->adj_expr * pow(base, power) * log(base);
  }
};

/// pow() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param base The base.
/// @param power The power.
template <typename Scalar>
ExpressionPtr<Scalar> pow(const ExpressionPtr<Scalar>& base,
                          const ExpressionPtr<Scalar>& power) {
  using enum ExpressionType;
  using std::pow;

  // Prune expression
  if (base->is_constant(Scalar(0))) {
    // Return zero, which base currently is
    return base;
  } else if (base->is_constant(Scalar(1))) {
    // Return one, which base currently is
    return base;
  }
  if (power->is_constant(Scalar(0))) {
    return constant_ptr(Scalar(1));
  } else if (power->is_constant(Scalar(1))) {
    // Return base unmodified
    return base;
  }

  // Evaluate constant
  if (base->type() == CONSTANT && power->type() == CONSTANT) {
    return constant_ptr(pow(base->val, power->val));
  }

  if (power->is_constant(Scalar(2))) {
    if (base->type() == LINEAR) {
      return make_expression_ptr<MultExpression<Scalar, QUADRATIC>>(base, base);
    } else {
      return make_expression_ptr<MultExpression<Scalar, NONLINEAR>>(base, base);
    }
  } else if (power->is_constant(Scalar(0.5))) {
    return sqrt(base);
  }

  return make_expression_ptr<PowExpression<Scalar, NONLINEAR>>(base, power);
}

/// Derived expression type for sign().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct SignExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr SignExpression(ExpressionPtr<Scalar> x)
      : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    if (x->val < Scalar(0)) {
      return Scalar(-1);
    } else if (x->val == Scalar(0)) {
      return Scalar(0);
    } else {
      return Scalar(1);
    }
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "sign"; }
};

/// sign() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> sign(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;

  // Evaluate constant
  if (x->type() == CONSTANT) {
    if (x->val < Scalar(0)) {
      return constant_ptr(Scalar(-1));
    } else if (x->val == Scalar(0)) {
      // Return zero
      return x;
    } else {
      return constant_ptr(Scalar(1));
    }
  }

  return make_expression_ptr<SignExpression<Scalar>>(x);
}

/// Derived expression type for sin().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct SinExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr SinExpression(ExpressionPtr<Scalar> x) : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::sin;
    return sin(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "sin"; }

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

 private:
  Scalar grad_l() const {
    using std::cos;
    return this->adj * cos(x->val);
  }

  ExpressionPtr<Scalar> grad_expr_l() const { return this->adj_expr * cos(x); }
};

/// sin() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> sin(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::sin;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    // Return zero, which x currently is
    return x;
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(sin(x->val));
  }

  return make_expression_ptr<SinExpression<Scalar>>(x);
}

/// Derived expression type for sinh().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct SinhExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr SinhExpression(ExpressionPtr<Scalar> x)
      : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::sinh;
    return sinh(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "sinh"; }

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

 private:
  Scalar grad_l() const {
    using std::cosh;
    return this->adj * cosh(x->val);
  }

  ExpressionPtr<Scalar> grad_expr_l() const { return this->adj_expr * cosh(x); }
};

/// sinh() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> sinh(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::sinh;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    // Return zero, which x currently is
    return x;
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(sinh(x->val));
  }

  return make_expression_ptr<SinhExpression<Scalar>>(x);
}

/// Derived expression type for sqrt().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct SqrtExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr SqrtExpression(ExpressionPtr<Scalar> x)
      : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::sqrt;
    return sqrt(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "sqrt"; }

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

 private:
  Scalar grad_l() const {
    using std::sqrt;
    return this->adj / (Scalar(2) * sqrt(x->val));
  }

  ExpressionPtr<Scalar> grad_expr_l() const {
    return this->adj_expr / (constant_ptr(Scalar(2)) * sqrt(x));
  }
};

/// sqrt() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> sqrt(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::sqrt;

  // Evaluate constant
  if (x->type() == CONSTANT) {
    if (x->val == Scalar(0)) {
      // Return zero
      return x;
    } else if (x->val == Scalar(1)) {
      return x;
    } else {
      return constant_ptr(sqrt(x->val));
    }
  }

  return make_expression_ptr<SqrtExpression<Scalar>>(x);
}

/// Derived expression type for tan().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct TanExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr TanExpression(ExpressionPtr<Scalar> x) : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::tan;
    return tan(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "tan"; }

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

 private:
  Scalar grad_l() const {
    using std::cos;

    auto c = cos(x->val);
    return this->adj / (c * c);
  }

  ExpressionPtr<Scalar> grad_expr_l() const {
    auto c = cos(x);
    return this->adj_expr / (c * c);
  }
};

/// tan() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> tan(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::tan;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    // Return zero, which x currently is
    return x;
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(tan(x->val));
  }

  return make_expression_ptr<TanExpression<Scalar>>(x);
}

/// Derived expression type for tanh().
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct TanhExpression final : Expression<Scalar> {
  /// Unary operator's operand.
  ExpressionPtr<Scalar> x;

  /// Constructs an unary expression (an operator with one argument).
  ///
  /// @param x Unary operator's operand.
  explicit constexpr TanhExpression(ExpressionPtr<Scalar> x)
      : x{std::move(x)} {}

  void visit_args(
      function_ref<void(Expression<Scalar>* arg)> func) const override {
    func(x.get());
  }

  Scalar value() const override {
    using std::tanh;
    return tanh(x->val);
  }

  ExpressionType type() const override { return ExpressionType::NONLINEAR; }

  std::string_view name() const override { return "tanh"; }

  void accumulate_adjoints() const override { x->adj += grad_l(); }

  void accumulate_adjoints_expr() const override {
    x->adj_expr += grad_expr_l();
  }

 private:
  Scalar grad_l() const {
    using std::cosh;

    auto c = cosh(x->val);
    return this->adj / (c * c);
  }

  ExpressionPtr<Scalar> grad_expr_l() const {
    auto c = cosh(x);
    return this->adj_expr / (c * c);
  }
};

/// tanh() for Expressions.
///
/// @tparam Scalar Scalar type.
/// @param x The argument.
template <typename Scalar>
ExpressionPtr<Scalar> tanh(const ExpressionPtr<Scalar>& x) {
  using enum ExpressionType;
  using std::tanh;

  // Prune expression
  if (x->is_constant(Scalar(0))) {
    // Return zero, which x currently is
    return x;
  }

  // Evaluate constant
  if (x->type() == CONSTANT) {
    return constant_ptr(tanh(x->val));
  }

  return make_expression_ptr<TanhExpression<Scalar>>(x);
}

}  // namespace slp::detail
