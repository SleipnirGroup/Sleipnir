// Copyright (c) Joshua Nichols and Tyler Veness

#include <gtest/gtest.h>
#include <sleipnir/autodiff/Jacobian.h>

TEST(JacobianTest, YvsX) {
  sleipnir::autodiff::VectorXvar y{3, 1};
  sleipnir::autodiff::VectorXvar x{3};
  x << 1, 2, 3;

  // y = x
  //
  //         [1  0  0]
  // dy/dx = [0  1  0]
  //         [0  0  1]
  y = x;
  Eigen::MatrixXd J = sleipnir::autodiff::Jacobian{y, x}.Calculate();

  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      if (row == col) {
        EXPECT_DOUBLE_EQ(1.0, J(row, col));
      } else {
        EXPECT_DOUBLE_EQ(0.0, J(row, col));
      }
    }
  }
}

TEST(JacobianTest, Yvs3X) {
  sleipnir::autodiff::VectorXvar y{3, 1};
  sleipnir::autodiff::VectorXvar x{3};
  x << 1, 2, 3;

  // y = 3x
  //
  //         [3  0  0]
  // dy/dx = [0  3  0]
  //         [0  0  3]
  y = 3 * x;
  Eigen::MatrixXd J = sleipnir::autodiff::Jacobian{y, x}.Calculate();

  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      if (row == col) {
        EXPECT_DOUBLE_EQ(3.0, J(row, col));
      } else {
        EXPECT_DOUBLE_EQ(0.0, J(row, col));
      }
    }
  }
}

TEST(JacobianTest, Products) {
  sleipnir::autodiff::VectorXvar y{3, 1};
  sleipnir::autodiff::VectorXvar x{3};
  x << 1, 2, 3;

  //     [x₁x₂]
  // y = [x₂x₃]
  //     [x₁x₃]
  //
  //         [x₂  x₁  0 ]
  // dy/dx = [0   x₃  x₂]
  //         [x₃  0   x₁]
  //
  //         [2  1  0]
  // dy/dx = [0  3  2]
  //         [3  0  1]
  y(0) = x(0) * x(1);
  y(1) = x(1) * x(2);
  y(2) = x(0) * x(2);
  Eigen::MatrixXd J = sleipnir::autodiff::Jacobian{y, x}.Calculate();

  EXPECT_DOUBLE_EQ(2.0, J(0, 0));
  EXPECT_DOUBLE_EQ(1.0, J(0, 1));
  EXPECT_DOUBLE_EQ(0.0, J(0, 2));
  EXPECT_DOUBLE_EQ(0.0, J(1, 0));
  EXPECT_DOUBLE_EQ(3.0, J(1, 1));
  EXPECT_DOUBLE_EQ(2.0, J(1, 2));
  EXPECT_DOUBLE_EQ(3.0, J(2, 0));
  EXPECT_DOUBLE_EQ(0.0, J(2, 1));
  EXPECT_DOUBLE_EQ(1.0, J(2, 2));
}

TEST(JacobianTest, DISABLED_NestedProducts) {
  sleipnir::autodiff::VectorXvar y{3, 1};
  sleipnir::autodiff::VectorXvar x{3};
  sleipnir::autodiff::Variable z = 1;
  x << 1 * z, 2 * z, 3 * z;

  //     [x₁x₂]
  // y = [x₂x₃]
  //     [x₁x₃]
  //
  //         [x₂  x₁  0 ]
  // dy/dx = [0   x₃  x₂]
  //         [x₃  0   x₁]
  //
  //         [2  1  0]
  // dy/dx = [0  3  2]
  //         [3  0  1]
  y(0) = x(0) * x(1);
  y(1) = x(1) * x(2);
  y(2) = x(0) * x(2);
  Eigen::MatrixXd J = sleipnir::autodiff::Jacobian{y, x}.Calculate();

  EXPECT_DOUBLE_EQ(2.0, J(0, 0));
  EXPECT_DOUBLE_EQ(1.0, J(0, 1));
  EXPECT_DOUBLE_EQ(0.0, J(0, 2));
  EXPECT_DOUBLE_EQ(0.0, J(1, 0));
  EXPECT_DOUBLE_EQ(3.0, J(1, 1));
  EXPECT_DOUBLE_EQ(2.0, J(1, 2));
  EXPECT_DOUBLE_EQ(3.0, J(2, 0));
  EXPECT_DOUBLE_EQ(0.0, J(2, 1));
  EXPECT_DOUBLE_EQ(1.0, J(2, 2));
}

TEST(JacobianTest, NonSquare) {
  sleipnir::autodiff::VectorXvar y{1, 1};
  sleipnir::autodiff::VectorXvar x{3};
  x << 1, 2, 3;

  // y = [x₁ + 3x₂ − 5x₃]
  //
  // dy/dx = [1  3  −5]
  y(0) = x(0) + 3 * x(1) - 5 * x(2);
  Eigen::MatrixXd J = sleipnir::autodiff::Jacobian{y, x}.Calculate();

  EXPECT_EQ(1, J.rows());
  EXPECT_EQ(3, J.cols());
  EXPECT_DOUBLE_EQ(1.0, J(0, 0));
  EXPECT_DOUBLE_EQ(3.0, J(0, 1));
  EXPECT_DOUBLE_EQ(-5.0, J(0, 2));
}
