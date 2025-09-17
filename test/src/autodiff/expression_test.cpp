// Copyright (c) Sleipnir contributors

#include <numbers>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/expression.hpp>

#include "catch_string_converters.hpp"

using slp::detail::ConstExpression;
using slp::detail::make_expression_ptr;

TEST_CASE("Expression - Default constructor", "[Expression]") {
  auto expr = make_expression_ptr<ConstExpression>();

  CHECK(expr->val == 0.0);
  CHECK(expr->type() == slp::ExpressionType::CONSTANT);
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

  CHECK(slp::detail::abs(zero)->is_constant(0.0));
  CHECK(slp::detail::abs(one)->is_constant(1.0));
}

TEST_CASE("Expression - Prune acos()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(slp::detail::acos(zero)->is_constant(std::numbers::pi / 2.0));
  CHECK(slp::detail::acos(one)->is_constant(std::acos(1.0)));
}

TEST_CASE("Expression - Prune asin()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(slp::detail::asin(zero)->is_constant(0.0));
  CHECK(slp::detail::asin(one)->is_constant(std::asin(1.0)));
}

TEST_CASE("Expression - Prune atan()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(slp::detail::atan(zero)->is_constant(0.0));
  CHECK(slp::detail::atan(one)->is_constant(std::atan(1.0)));
}

TEST_CASE("Expression - Prune atan2()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(slp::detail::atan2(zero, one)->is_constant(0.0));
  CHECK(slp::detail::atan2(one, zero)->is_constant(std::numbers::pi / 2.0));
  CHECK(slp::detail::atan2(one, one)->is_constant(std::atan2(1.0, 1.0)));
}

TEST_CASE("Expression - Prune cbrt()", "[Expression]") {
  auto negative_one = make_expression_ptr<ConstExpression>(-1.0);
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);
  auto two = make_expression_ptr<ConstExpression>(2.0);

  CHECK(slp::detail::cbrt(negative_one) == negative_one);
  CHECK(slp::detail::cbrt(zero)->is_constant(0.0));
  CHECK(slp::detail::cbrt(one) == one);

  auto c = slp::detail::cbrt(two);
  CHECK(c->type() == slp::ExpressionType::CONSTANT);
  CHECK(c->val == Catch::Approx(std::cbrt(2.0)).margin(1e-15));
}

TEST_CASE("Expression - Prune cos()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(slp::detail::cos(zero)->is_constant(1.0));
  CHECK(slp::detail::cos(one)->is_constant(std::cos(1.0)));
}

TEST_CASE("Expression - Prune cosh()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(slp::detail::cosh(zero)->is_constant(1.0));
  CHECK(slp::detail::cosh(one)->is_constant(std::cosh(1.0)));
}

TEST_CASE("Expression - Prune erf()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(slp::detail::erf(zero)->is_constant(0.0));
  CHECK(slp::detail::erf(one)->is_constant(std::erf(1.0)));
}

TEST_CASE("Expression - Prune exp()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(slp::detail::exp(zero)->is_constant(1.0));
  CHECK(slp::detail::exp(one)->is_constant(std::exp(1.0)));
}

TEST_CASE("Expression - Prune hypot()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(slp::detail::hypot(zero, zero)->is_constant(0.0));
  CHECK(slp::detail::hypot(zero, one) == one);
  CHECK(slp::detail::hypot(one, zero) == one);
  CHECK(slp::detail::hypot(one, one)->is_constant(std::numbers::sqrt2));
}

TEST_CASE("Expression - Prune log()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(slp::detail::log(zero)->is_constant(0.0));
  CHECK(slp::detail::log(one)->is_constant(std::log(1.0)));
}

TEST_CASE("Expression - Prune log10()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(slp::detail::log10(zero)->is_constant(0.0));
  CHECK(slp::detail::log10(one)->is_constant(std::log10(1.0)));
}

TEST_CASE("Expression - Prune pow()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);
  auto two = make_expression_ptr<ConstExpression>(2.0);

  CHECK(slp::detail::pow(zero, zero)->is_constant(0.0));
  CHECK(slp::detail::pow(zero, one)->is_constant(0.0));
  CHECK(slp::detail::pow(zero, two)->is_constant(0.0));
  CHECK(slp::detail::pow(one, zero) == one);
  CHECK(slp::detail::pow(one, one) == one);
  CHECK(slp::detail::pow(one, two) == one);
  CHECK(slp::detail::pow(two, zero)->is_constant(1.0));
  CHECK(slp::detail::pow(two, one) == two);
  CHECK(slp::detail::pow(two, two)->is_constant(4.0));
}

TEST_CASE("Expression - Prune sign()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);

  CHECK(slp::detail::sign(make_expression_ptr<ConstExpression>(-2.0))
            ->is_constant(-1.0));
  CHECK(slp::detail::sign(zero) == zero);
  CHECK(slp::detail::sign(make_expression_ptr<ConstExpression>(2.0))
            ->is_constant(1.0));
}

TEST_CASE("Expression - Prune sin()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(slp::detail::sin(zero)->is_constant(0.0));
  CHECK(slp::detail::sin(one)->is_constant(std::sin(1.0)));
}

TEST_CASE("Expression - Prune sinh()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(slp::detail::sinh(zero)->is_constant(0.0));
  CHECK(slp::detail::sinh(one)->is_constant(std::sinh(1.0)));
}

TEST_CASE("Expression - Prune sqrt()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);
  auto two = make_expression_ptr<ConstExpression>(2.0);

  CHECK(slp::detail::sqrt(zero)->is_constant(0.0));
  CHECK(slp::detail::sqrt(one) == one);
  CHECK(slp::detail::sqrt(two)->is_constant(std::numbers::sqrt2));
}

TEST_CASE("Expression - Prune tan()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(slp::detail::tan(zero)->is_constant(0.0));
  CHECK(slp::detail::tan(one)->is_constant(std::tan(1.0)));
}

TEST_CASE("Expression - Prune tanh()", "[Expression]") {
  auto zero = make_expression_ptr<ConstExpression>(0.0);
  auto one = make_expression_ptr<ConstExpression>(1.0);

  CHECK(slp::detail::tanh(zero)->is_constant(0.0));
  CHECK(slp::detail::tanh(one)->is_constant(std::tanh(1.0)));
}
