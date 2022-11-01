// Copyright (c) Joshua Nichols and Tyler Veness

#include <Eigen/Core>

#include "gtest/gtest.h"
#include "sleipnir/autodiff/Hessian.h"
#include "sleipnir/optimization/VariableMatrix.h"

TEST(VariableMatrixTest, HessianSumOfSquares) {
  sleipnir::autodiff::VectorXvar r{{25.0, 10.0, 5.0, 0.0}};
  sleipnir::autodiff::VectorXvar x{{0.0, 0.0, 0.0, 0.0}};

  sleipnir::VariableMatrix J = 0.0;
  for (int i = 0; i < 4; ++i) {
    J += (r(i) - x(i)) * (r(i) - x(i));
  }

  Eigen::MatrixXd H =
      sleipnir::autodiff::Hessian{J.Autodiff(0, 0), x}.Calculate();
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      if (row == col) {
        EXPECT_EQ(2.0, H(row, col));
      } else {
        EXPECT_EQ(0.0, H(row, col));
      }
    }
  }
}
