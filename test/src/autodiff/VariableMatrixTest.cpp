// Copyright (c) Sleipnir contributors

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <sleipnir/autodiff/VariableMatrix.hpp>
#include <sleipnir/optimization/Constraints.hpp>

TEST(VariableMatrixTest, AssignmentToDefault) {
  sleipnir::VariableMatrix mat;

  EXPECT_EQ(0, mat.Rows());
  EXPECT_EQ(0, mat.Cols());

  mat = sleipnir::VariableMatrix{2, 2};

  EXPECT_EQ(2, mat.Rows());
  EXPECT_EQ(2, mat.Cols());
  EXPECT_EQ(0.0, mat(0, 0));
  EXPECT_EQ(0.0, mat(0, 1));
  EXPECT_EQ(0.0, mat(1, 0));
  EXPECT_EQ(0.0, mat(1, 1));

  mat(0, 0) = 1.0;
  mat(0, 1) = 2.0;
  mat(1, 0) = 3.0;
  mat(1, 1) = 4.0;

  EXPECT_EQ(1.0, mat(0, 0));
  EXPECT_EQ(2.0, mat(0, 1));
  EXPECT_EQ(3.0, mat(1, 0));
  EXPECT_EQ(4.0, mat(1, 1));
}

TEST(VariableMatrixTest, BlockFreeFunction) {
  sleipnir::VariableMatrix A =
      Eigen::Matrix<double, 2, 3>{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  sleipnir::VariableMatrix B = Eigen::Matrix<double, 2, 1>{{7.0}, {8.0}};

  sleipnir::VariableMatrix mat1 = sleipnir::Block({{A, B}});
  Eigen::Matrix<double, 2, 4> expected1{{1.0, 2.0, 3.0, 7.0},
                                        {4.0, 5.0, 6.0, 8.0}};
  EXPECT_EQ(2, mat1.Rows());
  EXPECT_EQ(4, mat1.Cols());
  EXPECT_EQ(expected1, mat1.Value());

  sleipnir::VariableMatrix C =
      Eigen::Matrix<double, 1, 4>{{9.0, 10.0, 11.0, 12.0}};

  sleipnir::VariableMatrix mat2 = sleipnir::Block({{A, B}, {C}});
  Eigen::Matrix<double, 3, 4> expected2{
      {1.0, 2.0, 3.0, 7.0}, {4.0, 5.0, 6.0, 8.0}, {9.0, 10.0, 11.0, 12.0}};
  EXPECT_EQ(3, mat2.Rows());
  EXPECT_EQ(4, mat2.Cols());
  EXPECT_EQ(expected2, mat2.Value());
}
