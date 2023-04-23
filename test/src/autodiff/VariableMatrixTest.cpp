// Copyright (c) Sleipnir contributors

#include <gtest/gtest.h>
#include <sleipnir/autodiff/VariableMatrix.hpp>

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
