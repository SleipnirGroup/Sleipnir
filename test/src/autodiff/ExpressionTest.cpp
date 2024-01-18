// Copyright (c) Sleipnir contributors

#include <numbers>

#include <gtest/gtest.h>
#include <sleipnir/autodiff/Expression.hpp>

using sleipnir::detail::MakeExpressionPtr;
using sleipnir::detail::Zero;

TEST(ExpressionTest, DefaultConstructor) {
  auto expr = MakeExpressionPtr();

  EXPECT_EQ(0.0, expr->value);
  EXPECT_EQ(sleipnir::ExpressionType::kConstant, expr->type);
}

TEST(ExpressionTest, Zero) {
  EXPECT_TRUE(Zero()->IsConstant(0.0));
}

TEST(ExpressionTest, PruneMultiply) {
  auto zero = MakeExpressionPtr(0.0);
  auto one = MakeExpressionPtr(1.0);
  auto two = MakeExpressionPtr(2.0);

  EXPECT_EQ(zero * one, Zero());
  EXPECT_EQ(zero * two, Zero());
  EXPECT_EQ(one * zero, Zero());
  EXPECT_EQ(one * one, one);
  EXPECT_EQ(one * two, two);
  EXPECT_EQ(two * one, two);
}

TEST(ExpressionTest, PruneDivide) {
  auto zero = MakeExpressionPtr(0.0);
  auto one = MakeExpressionPtr(1.0);
  auto two = MakeExpressionPtr(2.0);

  EXPECT_EQ(zero / one, Zero());
  EXPECT_EQ(one / one, one);
  EXPECT_EQ(two / one, two);
}

TEST(ExpressionTest, PruneBinaryPlus) {
  auto zero = MakeExpressionPtr(0.0);
  auto one = MakeExpressionPtr(1.0);
  auto two = MakeExpressionPtr(2.0);

  EXPECT_EQ(zero + zero, zero);
  EXPECT_EQ(zero + one, one);
  EXPECT_EQ(zero + two, two);
  EXPECT_EQ(one + zero, one);
  EXPECT_EQ(two + zero, two);
}

TEST(ExpressionTest, PruneBinaryMinus) {
  auto zero = MakeExpressionPtr(0.0);
  auto one = MakeExpressionPtr(1.0);
  auto two = MakeExpressionPtr(2.0);

  EXPECT_EQ(zero - zero, Zero());
  EXPECT_EQ(one - zero, one);
  EXPECT_EQ(two - zero, two);
}

TEST(ExpressionTest, PruneUnaryPlus) {
  auto zero = MakeExpressionPtr(0.0);
  auto one = MakeExpressionPtr(1.0);
  auto two = MakeExpressionPtr(2.0);

  EXPECT_EQ(+zero, Zero());
  EXPECT_EQ(+one, one);
  EXPECT_EQ(+two, two);
}

TEST(ExpressionTest, PruneUnaryMinus) {
  auto zero = MakeExpressionPtr(0.0);

  EXPECT_EQ(-zero, Zero());
}

TEST(ExpressionTest, PruneAbs) {
  auto zero = MakeExpressionPtr(0.0);

  EXPECT_EQ(abs(zero), Zero());  // NOLINT
}

TEST(ExpressionTest, PruneAcos) {
  auto zero = MakeExpressionPtr(0.0);

  EXPECT_EQ(acos(zero)->value, std::numbers::pi / 2.0);  // NOLINT
}

TEST(ExpressionTest, PruneAsin) {
  auto zero = MakeExpressionPtr(0.0);

  EXPECT_EQ(asin(zero), Zero());  // NOLINT
}

TEST(ExpressionTest, PruneAtan) {
  auto zero = MakeExpressionPtr(0.0);

  EXPECT_EQ(atan(zero), Zero());  // NOLINT
}

TEST(ExpressionTest, PruneAtan2) {
  auto zero = MakeExpressionPtr(0.0);
  auto one = MakeExpressionPtr(1.0);

  EXPECT_EQ(atan2(zero, one), Zero());                         // NOLINT
  EXPECT_EQ(atan2(one, zero)->value, std::numbers::pi / 2.0);  // NOLINT
}

TEST(ExpressionTest, PruneCos) {
  auto zero = MakeExpressionPtr(0.0);

  EXPECT_EQ(cos(zero)->value, 1.0);  // NOLINT
}

TEST(ExpressionTest, PruneCosh) {
  auto zero = MakeExpressionPtr(0.0);

  EXPECT_EQ(cosh(zero)->value, 1.0);  // NOLINT
}

TEST(ExpressionTest, PruneErf) {
  auto zero = MakeExpressionPtr(0.0);

  EXPECT_EQ(erf(zero), Zero());  // NOLINT
}

TEST(ExpressionTest, PruneExp) {
  auto zero = MakeExpressionPtr(0.0);

  EXPECT_EQ(exp(zero)->value, 1.0);  // NOLINT
}

TEST(ExpressionTest, PruneHypot) {
  auto zero = MakeExpressionPtr(0.0);
  auto one = MakeExpressionPtr(1.0);

  EXPECT_EQ(hypot(zero, zero), Zero());  // NOLINT
  EXPECT_EQ(hypot(zero, one), one);      // NOLINT
  EXPECT_EQ(hypot(one, zero), one);      // NOLINT
}

TEST(ExpressionTest, PruneLog) {
  auto zero = MakeExpressionPtr(0.0);

  EXPECT_EQ(log(zero), Zero());  // NOLINT
}

TEST(ExpressionTest, PruneLog10) {
  auto zero = MakeExpressionPtr(0.0);

  EXPECT_EQ(log10(zero), Zero());  // NOLINT
}

TEST(ExpressionTest, PrunePow) {
  auto zero = MakeExpressionPtr(0.0);
  auto one = MakeExpressionPtr(1.0);
  auto two = MakeExpressionPtr(2.0);

  EXPECT_EQ(pow(zero, zero), Zero());     // NOLINT
  EXPECT_EQ(pow(zero, one), Zero());      // NOLINT
  EXPECT_EQ(pow(zero, two), Zero());      // NOLINT
  EXPECT_EQ(pow(one, zero), one);         // NOLINT
  EXPECT_EQ(pow(one, one), one);          // NOLINT
  EXPECT_EQ(pow(one, two), one);          // NOLINT
  EXPECT_EQ(pow(two, zero)->value, 1.0);  // NOLINT
  EXPECT_EQ(pow(two, one), two);          // NOLINT
}

TEST(ExpressionTest, PruneSign) {
  auto zero = MakeExpressionPtr(0.0);

  EXPECT_EQ(sign(zero), Zero());
}

TEST(ExpressionTest, PruneSin) {
  auto zero = MakeExpressionPtr(0.0);

  EXPECT_EQ(sin(zero), Zero());  // NOLINT
}

TEST(ExpressionTest, PruneSinh) {
  auto zero = MakeExpressionPtr(0.0);

  EXPECT_EQ(sinh(zero), Zero());
}

TEST(ExpressionTest, PruneSqrt) {
  auto zero = MakeExpressionPtr(0.0);
  auto one = MakeExpressionPtr(1.0);

  EXPECT_EQ(sqrt(zero), Zero());  // NOLINT
  EXPECT_EQ(sqrt(one), one);      // NOLINT
}

TEST(ExpressionTest, PruneTan) {
  auto zero = MakeExpressionPtr(0.0);

  EXPECT_EQ(tan(zero), Zero());  // NOLINT
}

TEST(ExpressionTest, PruneTanh) {
  auto zero = MakeExpressionPtr(0.0);

  EXPECT_EQ(tanh(zero), Zero());
}
