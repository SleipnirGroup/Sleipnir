// Copyright (c) Sleipnir contributors

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>

TEST(DecisionVariableTest, ScalarInitAssign) {
  sleipnir::OptimizationProblem problem;

  // Scalar zero init
  auto x = problem.DecisionVariable();
  EXPECT_DOUBLE_EQ(0.0, x.Value());

  // Scalar assignment
  x.SetValue(1.0);
  EXPECT_DOUBLE_EQ(1.0, x.Value());
  x.SetValue(2.0);
  EXPECT_DOUBLE_EQ(2.0, x.Value());
}

TEST(DecisionVariableTest, VectorInitAssign) {
  sleipnir::OptimizationProblem problem;

  // Vector zero init
  auto y = problem.DecisionVariable(2);
  EXPECT_DOUBLE_EQ(0.0, y.Value(0));
  EXPECT_DOUBLE_EQ(0.0, y.Value(1));

  // Vector assignment
  y(0).SetValue(1.0);
  y(1).SetValue(2.0);
  EXPECT_DOUBLE_EQ(1.0, y.Value(0));
  EXPECT_DOUBLE_EQ(2.0, y.Value(1));
  y(0).SetValue(3.0);
  y(1).SetValue(4.0);
  EXPECT_DOUBLE_EQ(3.0, y.Value(0));
  EXPECT_DOUBLE_EQ(4.0, y.Value(1));
}

TEST(DecisionVariableTest, StaticMatrixInitAssign) {
  sleipnir::OptimizationProblem problem;

  // Matrix zero init
  auto z = problem.DecisionVariable(3, 2);
  EXPECT_DOUBLE_EQ(0.0, z.Value(0, 0));
  EXPECT_DOUBLE_EQ(0.0, z.Value(0, 1));
  EXPECT_DOUBLE_EQ(0.0, z.Value(1, 0));
  EXPECT_DOUBLE_EQ(0.0, z.Value(1, 1));
  EXPECT_DOUBLE_EQ(0.0, z.Value(2, 0));
  EXPECT_DOUBLE_EQ(0.0, z.Value(2, 1));

  // Matrix assignment; element comparison
  z.SetValue(Eigen::Matrix<double, 3, 2>{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
  EXPECT_DOUBLE_EQ(1.0, z.Value(0, 0));
  EXPECT_DOUBLE_EQ(2.0, z.Value(0, 1));
  EXPECT_DOUBLE_EQ(3.0, z.Value(1, 0));
  EXPECT_DOUBLE_EQ(4.0, z.Value(1, 1));
  EXPECT_DOUBLE_EQ(5.0, z.Value(2, 0));
  EXPECT_DOUBLE_EQ(6.0, z.Value(2, 1));

  // Matrix assignment; matrix comparison
  {
    Eigen::Matrix<double, 3, 2> expected{{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
    z.SetValue(expected);
    EXPECT_EQ(expected, z.Value());
  }

  // Block assignment
  {
    Eigen::Matrix<double, 2, 1> expectedBlock{{1.0}, {1.0}};
    z.Block(0, 0, 2, 1).SetValue(expectedBlock);

    Eigen::Matrix<double, 3, 2> expectedResult{
        {1.0, 8.0}, {1.0, 10.0}, {11.0, 12.0}};
    EXPECT_EQ(expectedResult, z.Value());
  }

  // Segment assignment
  {
    Eigen::Matrix<double, 2, 1> expectedBlock{{1.0}, {1.0}};
    z.Segment(0, 2).SetValue(expectedBlock);

    Eigen::Matrix<double, 3, 2> expectedResult{
        {1.0, 8.0}, {1.0, 10.0}, {11.0, 12.0}};
    EXPECT_EQ(expectedResult, z.Value());
  }
}

TEST(DecisionVariableTest, DynamicMatrixInitAssign) {
  sleipnir::OptimizationProblem problem;

  // Matrix zero init
  auto z = problem.DecisionVariable(3, 2);
  EXPECT_DOUBLE_EQ(0.0, z.Value(0, 0));
  EXPECT_DOUBLE_EQ(0.0, z.Value(0, 1));
  EXPECT_DOUBLE_EQ(0.0, z.Value(1, 0));
  EXPECT_DOUBLE_EQ(0.0, z.Value(1, 1));
  EXPECT_DOUBLE_EQ(0.0, z.Value(2, 0));
  EXPECT_DOUBLE_EQ(0.0, z.Value(2, 1));

  // Matrix assignment; element comparison
  {
    Eigen::MatrixXd expected{3, 2};
    expected << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    z.SetValue(expected);
    EXPECT_DOUBLE_EQ(1.0, z.Value(0, 0));
    EXPECT_DOUBLE_EQ(2.0, z.Value(0, 1));
    EXPECT_DOUBLE_EQ(3.0, z.Value(1, 0));
    EXPECT_DOUBLE_EQ(4.0, z.Value(1, 1));
    EXPECT_DOUBLE_EQ(5.0, z.Value(2, 0));
    EXPECT_DOUBLE_EQ(6.0, z.Value(2, 1));
  }

  // Matrix assignment; matrix comparison
  {
    Eigen::MatrixXd expected{3, 2};
    expected << 7.0, 8.0, 9.0, 10.0, 11.0, 12.0;
    z.SetValue(expected);
    EXPECT_EQ(expected, z.Value());
  }

  // Block assignment
  {
    Eigen::MatrixXd expectedBlock{2, 1};
    expectedBlock << 1.0, 1.0;
    z.Block(0, 0, 2, 1).SetValue(expectedBlock);

    Eigen::MatrixXd expectedResult{3, 2};
    expectedResult << 1.0, 8.0, 1.0, 10.0, 11.0, 12.0;
    EXPECT_EQ(expectedResult, z.Value());
  }

  // Segment assignment
  {
    Eigen::MatrixXd expectedBlock{2, 1};
    expectedBlock << 1.0, 1.0;
    z.Segment(0, 2).SetValue(expectedBlock);

    Eigen::MatrixXd expectedResult{3, 2};
    expectedResult << 1.0, 8.0, 1.0, 10.0, 11.0, 12.0;
    EXPECT_EQ(expectedResult, z.Value());
  }
}

TEST(DecisionVariableTest, SymmetricMatrix) {
  sleipnir::OptimizationProblem problem;

  // Matrix zero init
  auto A = problem.SymmetricDecisionVariable(2);
  EXPECT_DOUBLE_EQ(0.0, A.Value(0, 0));
  EXPECT_DOUBLE_EQ(0.0, A.Value(0, 1));
  EXPECT_DOUBLE_EQ(0.0, A.Value(1, 0));
  EXPECT_DOUBLE_EQ(0.0, A.Value(1, 1));

  // Assign to lower triangle
  A(0, 0).SetValue(1.0);
  A(1, 0).SetValue(2.0);
  A(1, 1).SetValue(3.0);

  // Confirm whole matrix changed
  EXPECT_DOUBLE_EQ(1.0, A.Value(0, 0));
  EXPECT_DOUBLE_EQ(2.0, A.Value(0, 1));
  EXPECT_DOUBLE_EQ(2.0, A.Value(1, 0));
  EXPECT_DOUBLE_EQ(3.0, A.Value(1, 1));
}
