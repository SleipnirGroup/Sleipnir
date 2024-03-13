// Copyright (c) Sleipnir contributors

#include <numbers>

#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/Expression.hpp>

using sleipnir::detail::MakeExpressionPtr;

TEST_CASE("Expression - Default constructor", "[Expression]") {
  auto expr = MakeExpressionPtr();

  CHECK(expr->value == 0.0);
  CHECK(expr->type == sleipnir::ExpressionType::kConstant);
}

TEST_CASE("Expression - Zero", "[Expression]") {
  CHECK(MakeExpressionPtr()->IsConstant(0.0));
}

TEST_CASE("Expression - Prune multiply", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);
  auto one = MakeExpressionPtr(1.0);
  auto two = MakeExpressionPtr(2.0);

  CHECK((zero * one)->IsConstant(0.0));
  CHECK((zero * two)->IsConstant(0.0));
  CHECK((one * zero)->IsConstant(0.0));
  CHECK(one * one == one);
  CHECK(one * two == two);
  CHECK(two * one == two);
}

TEST_CASE("Expression - Prune divide", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);
  auto one = MakeExpressionPtr(1.0);
  auto two = MakeExpressionPtr(2.0);

  CHECK((zero / one)->IsConstant(0.0));
  CHECK(one / one == one);
  CHECK(two / one == two);
}

TEST_CASE("Expression - Prune binary plus", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);
  auto one = MakeExpressionPtr(1.0);
  auto two = MakeExpressionPtr(2.0);

  CHECK(zero + zero == zero);
  CHECK(zero + one == one);
  CHECK(zero + two == two);
  CHECK(one + zero == one);
  CHECK(two + zero == two);
}

TEST_CASE("Expression - Prune binary minus", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);
  auto one = MakeExpressionPtr(1.0);
  auto two = MakeExpressionPtr(2.0);

  CHECK((zero - zero)->IsConstant(0.0));
  CHECK(one - zero == one);
  CHECK(two - zero == two);
}

TEST_CASE("Expression - Prune unary plus", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);
  auto one = MakeExpressionPtr(1.0);
  auto two = MakeExpressionPtr(2.0);

  CHECK((+zero)->IsConstant(0.0));
  CHECK(+one == one);
  CHECK(+two == two);
}

TEST_CASE("Expression - Prune unary minus", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);

  CHECK((-zero)->IsConstant(0.0));
}

TEST_CASE("Expression - Prune abs()", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);

  CHECK(abs(zero)->IsConstant(0.0));  // NOLINT
}

TEST_CASE("Expression - Prune acos()", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);

  CHECK(acos(zero)->value == std::numbers::pi / 2.0);  // NOLINT
}

TEST_CASE("Expression - Prune asin()", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);

  CHECK(asin(zero)->IsConstant(0.0));  // NOLINT
}

TEST_CASE("Expression - Prune atan()", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);

  CHECK(atan(zero)->IsConstant(0.0));  // NOLINT
}

TEST_CASE("Expression - Prune atan2()", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);
  auto one = MakeExpressionPtr(1.0);

  CHECK(atan2(zero, one)->IsConstant(0.0));                  // NOLINT
  CHECK(atan2(one, zero)->value == std::numbers::pi / 2.0);  // NOLINT
}

TEST_CASE("Expression - Prune cos()", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);

  CHECK(cos(zero)->value == 1.0);  // NOLINT
}

TEST_CASE("Expression - Prune cosh()", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);

  CHECK(cosh(zero)->value == 1.0);  // NOLINT
}

TEST_CASE("Expression - Prune erf()", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);

  CHECK(erf(zero)->IsConstant(0.0));  // NOLINT
}

TEST_CASE("Expression - Prune exp()", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);

  CHECK(exp(zero)->value == 1.0);  // NOLINT
}

TEST_CASE("Expression - Prune hypot()", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);
  auto one = MakeExpressionPtr(1.0);

  CHECK(hypot(zero, zero)->IsConstant(0.0));  // NOLINT
  CHECK(hypot(zero, one) == one);             // NOLINT
  CHECK(hypot(one, zero) == one);             // NOLINT
}

TEST_CASE("Expression - Prune log()", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);

  CHECK(log(zero)->IsConstant(0.0));  // NOLINT
}

TEST_CASE("Expression - Prune log10()", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);

  CHECK(log10(zero)->IsConstant(0.0));  // NOLINT
}

TEST_CASE("Expression - Prune pow()", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);
  auto one = MakeExpressionPtr(1.0);
  auto two = MakeExpressionPtr(2.0);

  CHECK(pow(zero, zero)->IsConstant(0.0));  // NOLINT
  CHECK(pow(zero, one)->IsConstant(0.0));   // NOLINT
  CHECK(pow(zero, two)->IsConstant(0.0));   // NOLINT
  CHECK(pow(one, zero) == one);             // NOLINT
  CHECK(pow(one, one) == one);              // NOLINT
  CHECK(pow(one, two) == one);              // NOLINT
  CHECK(pow(two, zero)->value == 1.0);      // NOLINT
  CHECK(pow(two, one) == two);              // NOLINT
}

TEST_CASE("Expression - Prune sign()", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);

  CHECK(sign(zero)->IsConstant(0.0));
}

TEST_CASE("Expression - Prune sin()", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);

  CHECK(sin(zero)->IsConstant(0.0));  // NOLINT
}

TEST_CASE("Expression - Prune sinh()", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);

  CHECK(sinh(zero)->IsConstant(0.0));
}

TEST_CASE("Expression - Prune sqrt()", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);
  auto one = MakeExpressionPtr(1.0);

  CHECK(sqrt(zero)->IsConstant(0.0));  // NOLINT
  CHECK(sqrt(one) == one);             // NOLINT
}

TEST_CASE("Expression - Prune tan()", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);

  CHECK(tan(zero)->IsConstant(0.0));  // NOLINT
}

TEST_CASE("Expression - Prune tanh()", "[Expression]") {
  auto zero = MakeExpressionPtr(0.0);

  CHECK(tanh(zero)->IsConstant(0.0));
}
