// Copyright (c) Sleipnir contributors

#include <gtest/gtest.h>
#include <sleipnir/autodiff/Variable.hpp>

TEST(VariableTest, DefaultConstructor) {
  sleipnir::Variable a;

  EXPECT_EQ(0.0, a.Value());
  EXPECT_EQ(sleipnir::ExpressionType::kLinear, a.Type());
}
