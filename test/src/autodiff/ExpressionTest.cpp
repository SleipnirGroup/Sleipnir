// Copyright (c) Sleipnir contributors

#include <gtest/gtest.h>
#include <sleipnir/autodiff/Expression.hpp>

TEST(ExpressionTest, DefaultConstructor) {
  auto expr = sleipnir::detail::MakeExpressionPtr();

  EXPECT_EQ(0.0, expr->value);
  EXPECT_EQ(sleipnir::ExpressionType::kConstant, expr->type);
}

TEST(ExpressionTest, Zero) {
  EXPECT_TRUE(sleipnir::detail::Zero()->IsConstant(0.0));
}
