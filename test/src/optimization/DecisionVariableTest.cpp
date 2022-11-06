// Copyright (c) Joshua Nichols and Tyler Veness

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <sleipnir/optimization/OptimizationProblem.h>

TEST(DecisionVariableTest, ScalarInitAssign) {
  sleipnir::OptimizationProblem problem;

  // Scalar zero init
  auto x = problem.DecisionVariable();
  EXPECT_DOUBLE_EQ(0.0, x.Value(0));

  // Scalar assignment
  x = 1.0;
  EXPECT_DOUBLE_EQ(1.0, x.Value(0));
  x = 2.0;
  EXPECT_DOUBLE_EQ(2.0, x.Value(0));
}

TEST(DecisionVariableTest, VectorInitAssign) {
  sleipnir::OptimizationProblem problem;

  // Vector zero init
  auto y = problem.DecisionVariable(2);
  EXPECT_DOUBLE_EQ(0.0, y.Value(0));
  EXPECT_DOUBLE_EQ(0.0, y.Value(1));

  // Vector assignment
  y(0) = 1.0;
  y(1) = 2.0;
  EXPECT_DOUBLE_EQ(1.0, y.Value(0));
  EXPECT_DOUBLE_EQ(2.0, y.Value(1));
  y(0) = 3.0;
  y(1) = 4.0;
  EXPECT_DOUBLE_EQ(3.0, y.Value(0));
  EXPECT_DOUBLE_EQ(4.0, y.Value(1));
}

TEST(DecisionVariableTest, MatrixInitAssign) {
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
  z = Eigen::Matrix<double, 3, 2>{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  EXPECT_DOUBLE_EQ(1.0, z.Value(0, 0));
  EXPECT_DOUBLE_EQ(2.0, z.Value(0, 1));
  EXPECT_DOUBLE_EQ(3.0, z.Value(1, 0));
  EXPECT_DOUBLE_EQ(4.0, z.Value(1, 1));
  EXPECT_DOUBLE_EQ(5.0, z.Value(2, 0));
  EXPECT_DOUBLE_EQ(6.0, z.Value(2, 1));

  // Matrix assignment; matrix comparison
  {
    Eigen::Matrix<double, 3, 2> expected{{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
    z = expected;
    EXPECT_EQ(expected, z.Value());
  }

  // Block assignment
  {
    sleipnir::OptimizationProblem problem;

    Eigen::Matrix<double, 3, 2> expected{{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};

    Eigen::Matrix<double, 2, 1> expectedBlock{{1.0}, {1.0}};
    z.Block(0, 0, 2, 1) = expectedBlock;
    expected.block<2, 1>(0, 0) = expectedBlock;

    EXPECT_EQ(expected, z.Value());

    Eigen::Matrix<double, 3, 2> expectedResult{
        {1.0, 8.0}, {1.0, 10.0}, {11.0, 12.0}};
    EXPECT_EQ(expectedResult, z.Value());
  }

  // Segment assignment
  {
    sleipnir::OptimizationProblem problem;

    Eigen::Matrix<double, 3, 2> expected{{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};

    Eigen::Matrix<double, 2, 1> expectedBlock{{1.0}, {1.0}};
    z.Segment(0, 2) = expectedBlock;
    expected.block<2, 1>(0, 0) = expectedBlock;

    EXPECT_EQ(expected, z.Value());

    Eigen::Matrix<double, 3, 2> expectedResult{
        {1.0, 8.0}, {1.0, 10.0}, {11.0, 12.0}};
    EXPECT_EQ(expectedResult, z.Value());
  }
}
