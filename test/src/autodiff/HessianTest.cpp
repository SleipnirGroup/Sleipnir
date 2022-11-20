// Copyright (c) Joshua Nichols and Tyler Veness

#include <gtest/gtest.h>
#include <sleipnir/autodiff/Gradient.hpp>
#include <sleipnir/autodiff/Hessian.hpp>

TEST(HessianTest, Linear) {
  // y = x
  sleipnir::autodiff::VectorXvar x{1};
  x << 3;
  sleipnir::autodiff::Variable y = x(0);

  // dy/dx = 1
  double g = sleipnir::autodiff::Gradient{y, x(0)}.Calculate().coeff(0);
  EXPECT_DOUBLE_EQ(1.0, g);

  // d²y/dx² = d/dx(x (rhs) + x (lhs))
  //         = 1 + 1
  //         = 2
  Eigen::MatrixXd H = sleipnir::autodiff::Hessian{y, x}.Calculate();
  EXPECT_DOUBLE_EQ(0.0, H(0, 0));
}

TEST(HessianTest, Quadratic) {
  // y = x²
  // y = x * x
  sleipnir::autodiff::VectorXvar x{1};
  x << 3;
  sleipnir::autodiff::Variable y = x(0) * x(0);

  // dy/dx = x (rhs) + x (lhs)
  //       = (3) + (3)
  //       = 6
  double g = sleipnir::autodiff::Gradient{y, x(0)}.Calculate().coeff(0);
  EXPECT_DOUBLE_EQ(6.0, g);

  // d²y/dx² = d/dx(x (rhs) + x (lhs))
  //         = 1 + 1
  //         = 2
  Eigen::MatrixXd H = sleipnir::autodiff::Hessian{y, x}.Calculate();
  EXPECT_DOUBLE_EQ(2.0, H(0, 0));
}

TEST(HessianTest, Sum) {
  sleipnir::autodiff::Variable y;
  Eigen::VectorXd g;
  Eigen::MatrixXd H;
  sleipnir::autodiff::VectorXvar x{5};
  x << 1, 2, 3, 4, 5;

  // y = sum(x)
  y = x.sum();
  g = sleipnir::autodiff::Gradient{y, x}.Calculate();

  EXPECT_DOUBLE_EQ(15.0, y.Value());
  for (int i = 0; i < x.rows(); ++i) {
    EXPECT_DOUBLE_EQ(1.0, g(i));
  }

  H = sleipnir::autodiff::Hessian{y, x}.Calculate();
  for (int i = 0; i < x.rows(); ++i) {
    for (int j = 0; j < x.rows(); ++j) {
      EXPECT_DOUBLE_EQ(0.0, H(i, j));
    }
  }
}

TEST(HessianTest, SumOfProducts) {
  sleipnir::autodiff::Variable y;
  Eigen::VectorXd g;
  Eigen::MatrixXd H;
  sleipnir::autodiff::VectorXvar x{5};
  x << 1, 2, 3, 4, 5;

  // y = ||x||^2
  y = x.cwiseProduct(x).sum();
  g = sleipnir::autodiff::Gradient{y, x}.Calculate();

  EXPECT_DOUBLE_EQ(1 + 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5, y.Value());
  for (int i = 0; i < x.rows(); ++i) {
    EXPECT_EQ(2 * x(i), g(i));
  }

  H = sleipnir::autodiff::Hessian{y, x}.Calculate();
  for (int i = 0; i < x.rows(); ++i) {
    for (int j = 0; j < x.size(); ++j) {
      if (i == j) {
        EXPECT_DOUBLE_EQ(2.0, H(i, j));
      } else {
        EXPECT_DOUBLE_EQ(0.0, H(i, j));
      }
    }
  }
}

TEST(HessianTest, ProductOfSines) {
  sleipnir::autodiff::Variable y;
  Eigen::VectorXd g;
  Eigen::MatrixXd H;
  sleipnir::autodiff::VectorXvar x{5};
  x << 1, 2, 3, 4, 5;

  // y = prod(sin(x))
  y = x.array().sin().prod();
  g = sleipnir::autodiff::Gradient{y, x}.Calculate();

  EXPECT_EQ(sleipnir::autodiff::sin(1) * sleipnir::autodiff::sin(2) *
                sleipnir::autodiff::sin(3) * sleipnir::autodiff::sin(4) *
                sleipnir::autodiff::sin(5),
            y);
  for (int i = 0; i < x.rows(); ++i) {
    EXPECT_EQ(y / sleipnir::autodiff::tan(x(i)), g(i));
  }

  H = sleipnir::autodiff::Hessian{y, x}.Calculate();
  for (int i = 0; i < x.rows(); ++i) {
    for (int j = 0; j < x.rows(); ++j) {
      if (i == j) {
        EXPECT_NEAR((g(i) / sleipnir::autodiff::tan(x(i))).Value() *
                        (1.0 - 1.0 / (sleipnir::autodiff::cos(x(i)) *
                                      sleipnir::autodiff::cos(x(i))))
                            .Value(),
                    H(i, j), 1e-14);
      } else {
        EXPECT_NEAR((g(j) / sleipnir::autodiff::tan(x(i))).Value(), H(i, j),
                    1e-14);
      }
    }
  }
}

TEST(HessianTest, SumOfSquaredResiduals) {
  sleipnir::autodiff::Variable y;
  Eigen::VectorXd g;
  Eigen::MatrixXd H;
  sleipnir::autodiff::VectorXvar x{5};
  x << 1, 1, 1, 1, 1;

  // y = sum(diff(x).^2)
  y = (x.head(4) - x.tail(4)).array().pow(2).sum();
  g = sleipnir::autodiff::Gradient{y, x}.Calculate();

  EXPECT_EQ(0.0, y);
  EXPECT_EQ(2 * x[0] - 2 * x[1], g(0));
  EXPECT_EQ(-2 * x[0] + 4 * x[1] - 2 * x[2], g(1));
  EXPECT_EQ(-2 * x[1] + 4 * x[2] - 2 * x[3], g(2));
  EXPECT_EQ(-2 * x[2] + 4 * x[3] - 2 * x[4], g(3));
  EXPECT_EQ(-2 * x[3] + 2 * x[4], g(4));

  H = sleipnir::autodiff::Hessian{y, x}.Calculate();
  EXPECT_EQ(2.0, H(0, 0));
  EXPECT_EQ(-2.0, H(0, 1));
  EXPECT_EQ(0.0, H(0, 2));
  EXPECT_EQ(0.0, H(0, 3));
  EXPECT_EQ(0.0, H(0, 4));
  EXPECT_EQ(-2.0, H(1, 0));
  EXPECT_EQ(4.0, H(1, 1));
  EXPECT_EQ(-2.0, H(1, 2));
  EXPECT_EQ(0.0, H(1, 3));
  EXPECT_EQ(0.0, H(1, 4));
  EXPECT_EQ(0.0, H(2, 0));
  EXPECT_EQ(-2.0, H(2, 1));
  EXPECT_EQ(4.0, H(2, 2));
  EXPECT_EQ(-2.0, H(2, 3));
  EXPECT_EQ(0.0, H(2, 4));
  EXPECT_EQ(0.0, H(3, 0));
  EXPECT_EQ(0.0, H(3, 1));
  EXPECT_EQ(-2.0, H(3, 2));
  EXPECT_EQ(4.0, H(3, 3));
  EXPECT_EQ(-2.0, H(3, 4));
  EXPECT_EQ(0.0, H(4, 0));
  EXPECT_EQ(0.0, H(4, 1));
  EXPECT_EQ(0.0, H(4, 2));
  EXPECT_EQ(-2.0, H(4, 3));
  EXPECT_EQ(2.0, H(4, 4));
}

TEST(HessianTest, DISABLED_Reuse) {
  sleipnir::autodiff::Variable y;
  sleipnir::autodiff::VectorXvar x{1};

  // y = x³
  x << 1;
  y = x(0) * x(0) * x(0);

  sleipnir::autodiff::Hessian hessian{y, x};

  // d²y/dx² = 6x
  // H = 6
  Eigen::MatrixXd H = hessian.Calculate();

  EXPECT_EQ(1, H.rows());
  EXPECT_EQ(1, H.cols());
  EXPECT_DOUBLE_EQ(6.0, H(0, 0));

  x << 2;
  // d²y/dx² = 6x
  // H = 12
  H = hessian.Calculate();

  EXPECT_EQ(1, H.rows());
  EXPECT_EQ(1, H.cols());
  EXPECT_DOUBLE_EQ(12.0, H(0, 0));
}
