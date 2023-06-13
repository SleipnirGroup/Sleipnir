// Copyright (c) Sleipnir contributors

#include <gtest/gtest.h>
#include <sleipnir/autodiff/Jacobian.hpp>

TEST(JacobianTest, YvsX) {
  sleipnir::VectorXvar y{3};
  sleipnir::VectorXvar x{3};
  x(0) = 1;
  x(1) = 2;
  x(2) = 3;

  // y = x
  //
  //         [1  0  0]
  // dy/dx = [0  1  0]
  //         [0  0  1]
  y = x;
  Eigen::MatrixXd J = sleipnir::Jacobian{y, x}.Calculate();

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
  sleipnir::VectorXvar y{3};
  sleipnir::VectorXvar x{3};
  x(0) = 1;
  x(1) = 2;
  x(2) = 3;

  // y = 3x
  //
  //         [3  0  0]
  // dy/dx = [0  3  0]
  //         [0  0  3]
  y = 3 * x;
  Eigen::MatrixXd J = sleipnir::Jacobian{y, x}.Calculate();

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
  sleipnir::VectorXvar y{3};
  sleipnir::VectorXvar x{3};
  x(0) = 1;
  x(1) = 2;
  x(2) = 3;

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
  Eigen::MatrixXd J = sleipnir::Jacobian{y, x}.Calculate();

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
  sleipnir::VectorXvar z{1};
  z(0) = 1;
  sleipnir::VectorXvar x{3};
  x(0) = 1 * z(0);
  x(1) = 2 * z(0);
  x(2) = 3 * z(0);

  Eigen::MatrixXd J = sleipnir::Jacobian{x, z}.Calculate();
  EXPECT_DOUBLE_EQ(1.0, J(0, 0));
  EXPECT_DOUBLE_EQ(2.0, J(1, 0));
  EXPECT_DOUBLE_EQ(3.0, J(2, 0));

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
  sleipnir::VectorXvar y{3};
  y(0) = x(0) * x(1);
  y(1) = x(1) * x(2);
  y(2) = x(0) * x(2);
  J = sleipnir::Jacobian{y, x}.Calculate();

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
  sleipnir::VectorXvar y{1};
  sleipnir::VectorXvar x{3};
  x(0) = 1;
  x(1) = 2;
  x(2) = 3;

  // y = [x₁ + 3x₂ − 5x₃]
  //
  // dy/dx = [1  3  −5]
  y(0) = x(0) + 3 * x(1) - 5 * x(2);
  Eigen::MatrixXd J = sleipnir::Jacobian{y, x}.Calculate();

  EXPECT_EQ(1, J.rows());
  EXPECT_EQ(3, J.cols());
  EXPECT_DOUBLE_EQ(1.0, J(0, 0));
  EXPECT_DOUBLE_EQ(3.0, J(0, 1));
  EXPECT_DOUBLE_EQ(-5.0, J(0, 2));
}

TEST(JacobianTest, Reuse) {
  sleipnir::VectorXvar y{1};
  sleipnir::VectorXvar x{2};

  // y = [x₁x₂]
  x(0) = 1;
  x(1) = 2;
  y(0) = x(0) * x(1);

  sleipnir::Jacobian jacobian{y, x};

  // dy/dx = [x₂  x₁]
  // dy/dx = [2  1]
  Eigen::MatrixXd J = jacobian.Calculate();

  EXPECT_EQ(1, J.rows());
  EXPECT_EQ(2, J.cols());
  EXPECT_DOUBLE_EQ(2.0, J(0, 0));
  EXPECT_DOUBLE_EQ(1.0, J(0, 1));

  x(0) = 2;
  x(1) = 1;
  // dy/dx = [x₂  x₁]
  // dy/dx = [1  2]
  J = jacobian.Calculate();

  EXPECT_EQ(1, J.rows());
  EXPECT_EQ(2, J.cols());
  EXPECT_DOUBLE_EQ(1.0, J(0, 0));
  EXPECT_DOUBLE_EQ(2.0, J(0, 1));
}
