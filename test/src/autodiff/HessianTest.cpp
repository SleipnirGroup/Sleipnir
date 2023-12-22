// Copyright (c) Sleipnir contributors

#include <algorithm>
#include <functional>
#include <numeric>

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <sleipnir/autodiff/Gradient.hpp>
#include <sleipnir/autodiff/Hessian.hpp>
#include <sleipnir/autodiff/Variable.hpp>

#include "Range.hpp"

TEST(HessianTest, Linear) {
  // y = x
  sleipnir::VariableMatrix x{1};
  x(0).SetValue(3);
  sleipnir::Variable y = x(0);

  // dy/dx = 1
  double g = sleipnir::Gradient{y, x(0)}.Calculate().coeff(0);
  EXPECT_DOUBLE_EQ(1.0, g);

  // d²y/dx² = d/dx(x (rhs) + x (lhs))
  //         = 1 + 1
  //         = 2
  Eigen::MatrixXd H = sleipnir::Hessian{y, x}.Calculate();
  EXPECT_DOUBLE_EQ(0.0, H(0, 0));
}

TEST(HessianTest, Quadratic) {
  // y = x²
  // y = x * x
  sleipnir::VariableMatrix x{1};
  x(0).SetValue(3);
  sleipnir::Variable y = x(0) * x(0);

  // dy/dx = x (rhs) + x (lhs)
  //       = (3) + (3)
  //       = 6
  double g = sleipnir::Gradient{y, x(0)}.Calculate().coeff(0);
  EXPECT_DOUBLE_EQ(6.0, g);

  // d²y/dx² = d/dx(x (rhs) + x (lhs))
  //         = 1 + 1
  //         = 2
  Eigen::MatrixXd H = sleipnir::Hessian{y, x}.Calculate();
  EXPECT_DOUBLE_EQ(2.0, H(0, 0));
}

TEST(HessianTest, Sum) {
  sleipnir::Variable y;
  Eigen::VectorXd g;
  Eigen::MatrixXd H;
  sleipnir::VariableMatrix x{5};
  x(0).SetValue(1);
  x(1).SetValue(2);
  x(2).SetValue(3);
  x(3).SetValue(4);
  x(4).SetValue(5);

  // y = sum(x)
  y = std::accumulate(x.begin(), x.end(), sleipnir::Variable{0.0});
  g = sleipnir::Gradient{y, x}.Calculate();

  EXPECT_DOUBLE_EQ(15.0, y.Value());
  for (int i = 0; i < x.Rows(); ++i) {
    EXPECT_DOUBLE_EQ(1.0, g(i));
  }

  H = sleipnir::Hessian{y, x}.Calculate();
  for (int i = 0; i < x.Rows(); ++i) {
    for (int j = 0; j < x.Rows(); ++j) {
      EXPECT_DOUBLE_EQ(0.0, H(i, j));
    }
  }
}

TEST(HessianTest, SumOfProducts) {
  sleipnir::Variable y;
  Eigen::VectorXd g;
  Eigen::MatrixXd H;
  sleipnir::VariableMatrix x{5};
  x(0).SetValue(1);
  x(1).SetValue(2);
  x(2).SetValue(3);
  x(3).SetValue(4);
  x(4).SetValue(5);

  // y = ||x||²
  y = x.T() * x;
  g = sleipnir::Gradient{y, x}.Calculate();

  EXPECT_DOUBLE_EQ(1 + 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5, y.Value());
  for (int i = 0; i < x.Rows(); ++i) {
    EXPECT_DOUBLE_EQ((2 * x(i)).Value(), g(i));
  }

  H = sleipnir::Hessian{y, x}.Calculate();
  for (int i = 0; i < x.Rows(); ++i) {
    for (int j = 0; j < x.Rows(); ++j) {
      if (i == j) {
        EXPECT_DOUBLE_EQ(2.0, H(i, j));
      } else {
        EXPECT_DOUBLE_EQ(0.0, H(i, j));
      }
    }
  }
}

TEST(HessianTest, ProductOfSines) {
  sleipnir::Variable y;
  Eigen::VectorXd g;
  Eigen::MatrixXd H;
  sleipnir::VariableMatrix x{5};
  x(0).SetValue(1);
  x(1).SetValue(2);
  x(2).SetValue(3);
  x(3).SetValue(4);
  x(4).SetValue(5);

  // y = prod(sin(x))
  auto temp = x.CwiseTransform(sleipnir::sin);
  y = std::accumulate(temp.begin(), temp.end(), sleipnir::Variable{1.0},
                      std::multiplies{});
  g = sleipnir::Gradient{y, x}.Calculate();

  EXPECT_DOUBLE_EQ(
      std::sin(1) * std::sin(2) * std::sin(3) * std::sin(4) * std::sin(5),
      y.Value());
  for (int i = 0; i < x.Rows(); ++i) {
    EXPECT_DOUBLE_EQ((y / sleipnir::tan(x(i))).Value(), g(i));
  }

  H = sleipnir::Hessian{y, x}.Calculate();
  for (int i = 0; i < x.Rows(); ++i) {
    for (int j = 0; j < x.Rows(); ++j) {
      if (i == j) {
        EXPECT_NEAR(
            (g(i) / tan(x(i))).Value() *
                (1.0 - 1.0 / (sleipnir::cos(x(i)) * sleipnir::cos(x(i))))
                    .Value(),
            H(i, j), 1e-14);
      } else {
        EXPECT_DOUBLE_EQ((g(j) / tan(x(i))).Value(), H(i, j));
      }
    }
  }
}

TEST(HessianTest, SumOfSquaredResiduals) {
  sleipnir::Variable y;
  Eigen::VectorXd g;
  Eigen::MatrixXd H;
  sleipnir::VariableMatrix x{5};
  x(0).SetValue(1);
  x(1).SetValue(1);
  x(2).SetValue(1);
  x(3).SetValue(1);
  x(4).SetValue(1);

  // y = sum(diff(x).^2)
  auto temp = (x.Block(0, 0, 4, 1) - x.Block(1, 0, 4, 1))
                  .CwiseTransform([](const sleipnir::Variable& x) {
                    return sleipnir::pow(x, 2);
                  });
  y = std::accumulate(temp.begin(), temp.end(), sleipnir::Variable{0.0});
  g = sleipnir::Gradient{y, x}.Calculate();

  EXPECT_DOUBLE_EQ(0.0, y.Value());
  EXPECT_DOUBLE_EQ((2 * x(0) - 2 * x(1)).Value(), g(0));
  EXPECT_DOUBLE_EQ((-2 * x(0) + 4 * x(1) - 2 * x(2)).Value(), g(1));
  EXPECT_DOUBLE_EQ((-2 * x(1) + 4 * x(2) - 2 * x(3)).Value(), g(2));
  EXPECT_DOUBLE_EQ((-2 * x(2) + 4 * x(3) - 2 * x(4)).Value(), g(3));
  EXPECT_DOUBLE_EQ((-2 * x(3) + 2 * x(4)).Value(), g(4));

  H = sleipnir::Hessian{y, x}.Calculate();
  EXPECT_DOUBLE_EQ(2.0, H(0, 0));
  EXPECT_DOUBLE_EQ(-2.0, H(0, 1));
  EXPECT_DOUBLE_EQ(0.0, H(0, 2));
  EXPECT_DOUBLE_EQ(0.0, H(0, 3));
  EXPECT_DOUBLE_EQ(0.0, H(0, 4));
  EXPECT_DOUBLE_EQ(-2.0, H(1, 0));
  EXPECT_DOUBLE_EQ(4.0, H(1, 1));
  EXPECT_DOUBLE_EQ(-2.0, H(1, 2));
  EXPECT_DOUBLE_EQ(0.0, H(1, 3));
  EXPECT_DOUBLE_EQ(0.0, H(1, 4));
  EXPECT_DOUBLE_EQ(0.0, H(2, 0));
  EXPECT_DOUBLE_EQ(-2.0, H(2, 1));
  EXPECT_DOUBLE_EQ(4.0, H(2, 2));
  EXPECT_DOUBLE_EQ(-2.0, H(2, 3));
  EXPECT_DOUBLE_EQ(0.0, H(2, 4));
  EXPECT_DOUBLE_EQ(0.0, H(3, 0));
  EXPECT_DOUBLE_EQ(0.0, H(3, 1));
  EXPECT_DOUBLE_EQ(-2.0, H(3, 2));
  EXPECT_DOUBLE_EQ(4.0, H(3, 3));
  EXPECT_DOUBLE_EQ(-2.0, H(3, 4));
  EXPECT_DOUBLE_EQ(0.0, H(4, 0));
  EXPECT_DOUBLE_EQ(0.0, H(4, 1));
  EXPECT_DOUBLE_EQ(0.0, H(4, 2));
  EXPECT_DOUBLE_EQ(-2.0, H(4, 3));
  EXPECT_DOUBLE_EQ(2.0, H(4, 4));
}

TEST(HessianTest, SumOfSquares) {
  sleipnir::VariableMatrix r{4};
  r(0).SetValue(25.0);
  r(1).SetValue(10.0);
  r(2).SetValue(5.0);
  r(3).SetValue(0.0);
  sleipnir::VariableMatrix x{4};
  x(0).SetValue(0.0);
  x(1).SetValue(0.0);
  x(2).SetValue(0.0);
  x(3).SetValue(0.0);

  sleipnir::Variable J = 0.0;
  for (int i = 0; i < 4; ++i) {
    J += (r(i) - x(i)) * (r(i) - x(i));
  }

  Eigen::MatrixXd H = sleipnir::Hessian{J, x}.Calculate();
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      if (row == col) {
        EXPECT_DOUBLE_EQ(2.0, H(row, col));
      } else {
        EXPECT_DOUBLE_EQ(0.0, H(row, col));
      }
    }
  }
}

TEST(HessianTest, Rosenbrock) {
  sleipnir::VariableMatrix input{2};
  auto& x = input(0);
  auto& y = input(1);

  for (auto x0 : Range(-2.5, 2.5, 0.1)) {
    for (auto y0 : Range(-2.5, 2.5, 0.1)) {
      x.SetValue(x0);
      y.SetValue(y0);
      auto z = sleipnir::pow(1 - x, 2) +
               100 * sleipnir::pow(y - sleipnir::pow(x, 2), 2);

      Eigen::MatrixXd H = sleipnir::Hessian{z, input}.Calculate();
      EXPECT_DOUBLE_EQ(-400 * (y0 - x0 * x0) + 800 * x0 * x0 + 2, H(0, 0));
      EXPECT_DOUBLE_EQ(-400 * x0, H(0, 1));
      EXPECT_DOUBLE_EQ(-400 * x0, H(1, 0));
      EXPECT_DOUBLE_EQ(200, H(1, 1));
    }
  }
}

TEST(HessianTest, Reuse) {
  sleipnir::Variable y;
  sleipnir::VariableMatrix x{1};

  // y = x³
  x(0).SetValue(1);
  y = x(0) * x(0) * x(0);

  sleipnir::Hessian hessian{y, x};

  // d²y/dx² = 6x
  // H = 6
  Eigen::MatrixXd H = hessian.Calculate();

  EXPECT_EQ(1, H.rows());
  EXPECT_EQ(1, H.cols());
  EXPECT_DOUBLE_EQ(6.0, H(0, 0));

  x(0).SetValue(2);
  // d²y/dx² = 6x
  // H = 12
  H = hessian.Calculate();

  EXPECT_EQ(1, H.rows());
  EXPECT_EQ(1, H.cols());
  EXPECT_DOUBLE_EQ(12.0, H(0, 0));
}
