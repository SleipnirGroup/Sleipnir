// Copyright (c) Sleipnir contributors

#include <numbers>

#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/expression.hpp>

#include "catch_string_converters.hpp"

using sleipnir::detail::ConstExpression;
using sleipnir::detail::make_expression_ptr;

TEST_CASE("Expression - Default constructor", "[Expression]") {
  auto expr = make_expression_ptr<ConstExpression>();

  CHECK(expr->val == 0.0);
  CHECK(expr->type() == sleipnir::ExpressionType::CONSTANT);
}

TEST_CASE("Expression - Zero", "[Expression]") {
  CHECK(make_expression_ptr<ConstExpression>()->is_constant(0.0));
}

TEST_CASE("Expression - Prune multiply", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);
  auto two = make_expression_ptr<ConstExpression>(2.0);

  CHECK((zero * one)->is_constant(0.0));
  CHECK((zero * two)->is_constant(0.0));
  CHECK((one * zero)->is_constant(0.0));
  CHECK(one * one == one);
  CHECK(one * two == two);
  CHECK(two * one == two);
}

TEST_CASE("Expression - Prune divide", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);
  auto two = make_expression_ptr<ConstExpression>(2.0);

  CHECK((zero / one)->is_constant(0.0));
  CHECK(one / one == one);
  CHECK(two / one == two);
}

TEST_CASE("Expression - Prune binary plus", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);
  auto two = make_expression_ptr<ConstExpression>(2.0);

  CHECK(zero + zero == zero);
  CHECK(zero + one == one);
  CHECK(zero + two == two);
  CHECK(one + zero == one);
  CHECK(two + zero == two);
}

TEST_CASE("Expression - Prune binary minus", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);
  auto two = make_expression_ptr<ConstExpression>(2.0);

  CHECK((zero - zero)->is_constant(0.0));
  CHECK(one - zero == one);
  CHECK(two - zero == two);
}

TEST_CASE("Expression - Prune unary plus", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);
  auto two = make_expression_ptr<ConstExpression>(2.0);

  CHECK((+zero)->is_constant(0.0));
  CHECK(+one == one);
  CHECK(+two == two);
}

TEST_CASE("Expression - Prune unary minus", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);

  CHECK((-zero)->is_constant(0.0));
}

TEST_CASE("Expression - Prune abs()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(abs(zero)->is_constant(0.0));  // NOLINT
  CHECK(abs(one)->is_constant(1.0));   // NOLINT
}

TEST_CASE("Expression - Prune acos()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(acos(zero)->is_constant(std::numbers::pi / 2.0));  // NOLINT
  CHECK(acos(one)->is_constant(std::acos(1.0)));           // NOLINT
}

TEST_CASE("Expression - Prune asin()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(asin(zero)->is_constant(0.0));            // NOLINT
  CHECK(asin(one)->is_constant(std::asin(1.0)));  // NOLINT
}

TEST_CASE("Expression - Prune atan()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(atan(zero)->is_constant(0.0));            // NOLINT
  CHECK(atan(one)->is_constant(std::atan(1.0)));  // NOLINT
}

TEST_CASE("Expression - Prune atan2()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(atan2(zero, one)->is_constant(0.0));                     // NOLINT
  CHECK(atan2(one, zero)->is_constant(std::numbers::pi / 2.0));  // NOLINT
  CHECK(atan2(one, one)->is_constant(std::atan2(1.0, 1.0)));     // NOLINT
}

TEST_CASE("Expression - Prune cos()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(cos(zero)->is_constant(1.0));           // NOLINT
  CHECK(cos(one)->is_constant(std::cos(1.0)));  // NOLINT
}

TEST_CASE("Expression - Prune cosh()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(cosh(zero)->is_constant(1.0));            // NOLINT
  CHECK(cosh(one)->is_constant(std::cosh(1.0)));  // NOLINT
}

TEST_CASE("Expression - Prune erf()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(erf(zero)->is_constant(0.0));           // NOLINT
  CHECK(erf(one)->is_constant(std::erf(1.0)));  // NOLINT
}

TEST_CASE("Expression - Prune exp()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(exp(zero)->is_constant(1.0));           // NOLINT
  CHECK(exp(one)->is_constant(std::exp(1.0)));  // NOLINT
}

TEST_CASE("Expression - Prune hypot()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(hypot(zero, zero)->is_constant(0.0));                // NOLINT
  CHECK(hypot(zero, one) == one);                            // NOLINT
  CHECK(hypot(one, zero) == one);                            // NOLINT
  CHECK(hypot(one, one)->is_constant(std::numbers::sqrt2));  // NOLINT
}

TEST_CASE("Expression - Prune log()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(log(zero)->is_constant(0.0));           // NOLINT
  CHECK(log(one)->is_constant(std::log(1.0)));  // NOLINT
}

TEST_CASE("Expression - Prune log10()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(log10(zero)->is_constant(0.0));             // NOLINT
  CHECK(log10(one)->is_constant(std::log10(1.0)));  // NOLINT
}

TEST_CASE("Expression - Prune pow()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);
  auto two = make_expression_ptr<ConstExpression>(2.0);

  CHECK(pow(zero, zero)->is_constant(0.0));  // NOLINT
  CHECK(pow(zero, one)->is_constant(0.0));   // NOLINT
  CHECK(pow(zero, two)->is_constant(0.0));   // NOLINT
  CHECK(pow(one, zero) == one);              // NOLINT
  CHECK(pow(one, one) == one);               // NOLINT
  CHECK(pow(one, two) == one);               // NOLINT
  CHECK(pow(two, zero)->is_constant(1.0));   // NOLINT
  CHECK(pow(two, one) == two);               // NOLINT
  CHECK(pow(two, two)->is_constant(4.0));    // NOLINT
}

TEST_CASE("Expression - Prune sign()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);

  CHECK(sign(make_expression_ptr<ConstExpression>(-2.0))->is_constant(-1.0));
  CHECK(sign(zero) == zero);
  CHECK(sign(make_expression_ptr<ConstExpression>(2.0))->is_constant(1.0));
}

TEST_CASE("Expression - Prune sin()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(sin(zero)->is_constant(0.0));           // NOLINT
  CHECK(sin(one)->is_constant(std::sin(1.0)));  // NOLINT
}

TEST_CASE("Expression - Prune sinh()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(sinh(zero)->is_constant(0.0));
  CHECK(sinh(one)->is_constant(std::sinh(1.0)));
}

TEST_CASE("Expression - Prune sqrt()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);
  auto two = make_expression_ptr<ConstExpression>(2.0);

  CHECK(sqrt(zero)->is_constant(0.0));                 // NOLINT
  CHECK(sqrt(one) == one);                             // NOLINT
  CHECK(sqrt(two)->is_constant(std::numbers::sqrt2));  // NOLINT
}

TEST_CASE("Expression - Prune tan()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(tan(zero)->is_constant(0.0));           // NOLINT
  CHECK(tan(one)->is_constant(std::tan(1.0)));  // NOLINT
}

TEST_CASE("Expression - Prune tanh()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(tanh(zero)->is_constant(0.0));
  CHECK(tanh(one)->is_constant(std::tanh(1.0)));
}
