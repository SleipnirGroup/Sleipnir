// Copyright (c) Sleipnir contributors

#include <gtest/gtest.h>
#include <sleipnir/autodiff/Variable.hpp>
#include <sleipnir/optimization/Constraints.hpp>

TEST(ConstraintsTest, EqualityConstraintBooleanComparisons) {
  EXPECT_TRUE(sleipnir::Variable{1.0} == sleipnir::Variable{1.0});
  EXPECT_FALSE(sleipnir::Variable{1.0} == sleipnir::Variable{2.0});
}

TEST(ConstraintsTest, InequalityConstraintBooleanComparisons) {
  // These are all true because for the purposes of optimization, a < constraint
  // is treated the same as a <= constraint
  EXPECT_TRUE(sleipnir::Variable{1.0} < sleipnir::Variable{1.0});
  EXPECT_TRUE(sleipnir::Variable{1.0} <= sleipnir::Variable{1.0});
  EXPECT_TRUE(sleipnir::Variable{1.0} > sleipnir::Variable{1.0});
  EXPECT_TRUE(sleipnir::Variable{1.0} >= sleipnir::Variable{1.0});

  EXPECT_TRUE(sleipnir::Variable{1.0} < sleipnir::Variable{2.0});
  EXPECT_TRUE(sleipnir::Variable{1.0} <= sleipnir::Variable{2.0});
  EXPECT_FALSE(sleipnir::Variable{1.0} > sleipnir::Variable{2.0});
  EXPECT_FALSE(sleipnir::Variable{1.0} >= sleipnir::Variable{2.0});

  EXPECT_FALSE(sleipnir::Variable{2.0} < sleipnir::Variable{1.0});
  EXPECT_FALSE(sleipnir::Variable{2.0} <= sleipnir::Variable{1.0});
  EXPECT_TRUE(sleipnir::Variable{2.0} > sleipnir::Variable{1.0});
  EXPECT_TRUE(sleipnir::Variable{2.0} >= sleipnir::Variable{1.0});
}
