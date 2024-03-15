// Copyright (c) Sleipnir contributors

#include <cmath>
#include <functional>
#include <numeric>

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/Gradient.hpp>
#include <sleipnir/autodiff/Hessian.hpp>
#include <sleipnir/autodiff/Variable.hpp>

#include "Range.hpp"

TEST_CASE("Hessian - Linear", "[Hessian]") {
  // y = x
  sleipnir::VariableMatrix x{1};
  x(0).SetValue(3);
  sleipnir::Variable y = x(0);

  // dy/dx = 1
  double g = sleipnir::Gradient(y, x(0)).Value().coeff(0);
  CHECK(g == 1.0);

  // d²y/dx² = 0
  Eigen::MatrixXd H = sleipnir::Hessian(y, x).Value();
  CHECK(H(0, 0) == 0.0);
}

TEST_CASE("Hessian - Quadratic", "[Hessian]") {
  // y = x²
  // y = x * x
  sleipnir::VariableMatrix x{1};
  x(0).SetValue(3);
  sleipnir::Variable y = x(0) * x(0);

  // dy/dx = x (rhs) + x (lhs)
  //       = (3) + (3)
  //       = 6
  double g = sleipnir::Gradient(y, x(0)).Value().coeff(0);
  CHECK(g == 6.0);

  // d²y/dx² = d/dx(x (rhs) + x (lhs))
  //         = 1 + 1
  //         = 2
  Eigen::MatrixXd H = sleipnir::Hessian(y, x).Value();
  CHECK(H(0, 0) == 2.0);
}

TEST_CASE("Hessian - Sum", "[Hessian]") {
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
  g = sleipnir::Gradient(y, x).Value();

  CHECK(y.Value() == 15.0);
  for (int i = 0; i < x.Rows(); ++i) {
    CHECK(g(i) == 1.0);
  }

  H = sleipnir::Hessian(y, x).Value();
  for (int i = 0; i < x.Rows(); ++i) {
    for (int j = 0; j < x.Rows(); ++j) {
      CHECK(H(i, j) == 0.0);
    }
  }
}

TEST_CASE("Hessian - Sum of products", "[Hessian]") {
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
  g = sleipnir::Gradient(y, x).Value();

  CHECK(y.Value() == 1 + 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5);
  for (int i = 0; i < x.Rows(); ++i) {
    CHECK(g(i) == (2 * x(i)).Value());
  }

  H = sleipnir::Hessian(y, x).Value();
  for (int i = 0; i < x.Rows(); ++i) {
    for (int j = 0; j < x.Rows(); ++j) {
      if (i == j) {
        CHECK(H(i, j) == 2.0);
      } else {
        CHECK(H(i, j) == 0.0);
      }
    }
  }
}

TEST_CASE("Hessian - Product of sines", "[Hessian]") {
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
  g = sleipnir::Gradient(y, x).Value();

  CHECK(y.Value() ==
        std::sin(1) * std::sin(2) * std::sin(3) * std::sin(4) * std::sin(5));
  for (int i = 0; i < x.Rows(); ++i) {
    CHECK(g(i) ==
          Catch::Approx((y / sleipnir::tan(x(i))).Value()).margin(1e-15));
  }

  H = sleipnir::Hessian(y, x).Value();
  for (int i = 0; i < x.Rows(); ++i) {
    for (int j = 0; j < x.Rows(); ++j) {
      if (i == j) {
        CHECK(H(i, j) == Catch::Approx((g(i) / sleipnir::tan(x(i))).Value() *
                                       (1.0 - 1.0 / (sleipnir::cos(x(i)) *
                                                     sleipnir::cos(x(i))))
                                           .Value())
                             .margin(1e-15));
      } else {
        CHECK(
            H(i, j) ==
            Catch::Approx((g(j) / sleipnir::tan(x(i))).Value()).margin(1e-15));
      }
    }
  }
}

TEST_CASE("Hessian - Sum of squared residuals", "[Hessian]") {
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
  g = sleipnir::Gradient(y, x).Value();

  CHECK(y.Value() == 0.0);
  CHECK(g(0) == (2 * x(0) - 2 * x(1)).Value());
  CHECK(g(1) == (-2 * x(0) + 4 * x(1) - 2 * x(2)).Value());
  CHECK(g(2) == (-2 * x(1) + 4 * x(2) - 2 * x(3)).Value());
  CHECK(g(3) == (-2 * x(2) + 4 * x(3) - 2 * x(4)).Value());
  CHECK(g(4) == (-2 * x(3) + 2 * x(4)).Value());

  H = sleipnir::Hessian(y, x).Value();

  Eigen::MatrixXd expectedH{{2.0, -2.0, 0.0, 0.0, 0.0},
                            {-2.0, 4.0, -2.0, 0.0, 0.0},
                            {0.0, -2.0, 4.0, -2.0, 0.0},
                            {0.0, 0.0, -2.0, 4.0, -2.0},
                            {0.0, 0.0, 0.0, -2.0, 2.0}};
  for (int i = 0; i < x.Rows(); ++i) {
    for (int j = 0; j < x.Rows(); ++j) {
      CHECK(H(i, j) == expectedH(i, j));
    }
  }
}

TEST_CASE("Hessian - Sum of squares", "[Hessian]") {
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

  Eigen::MatrixXd H = sleipnir::Hessian(J, x).Value();
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      if (row == col) {
        CHECK(H(row, col) == 2.0);
      } else {
        CHECK(H(row, col) == 0.0);
      }
    }
  }
}

TEST_CASE("Hessian - Rosenbrock", "[Hessian]") {
  sleipnir::VariableMatrix input{2};
  auto& x = input(0);
  auto& y = input(1);

  for (auto x0 : Range(-2.5, 2.5, 0.1)) {
    for (auto y0 : Range(-2.5, 2.5, 0.1)) {
      x.SetValue(x0);
      y.SetValue(y0);
      auto z = sleipnir::pow(1 - x, 2) +
               100 * sleipnir::pow(y - sleipnir::pow(x, 2), 2);

      Eigen::MatrixXd H = sleipnir::Hessian(z, input).Value();
      CHECK(H(0, 0) == Catch::Approx(-400 * (y0 - x0 * x0) + 800 * x0 * x0 + 2)
                           .margin(1e-12));
      CHECK(H(0, 1) == -400 * x0);
      CHECK(H(1, 0) == -400 * x0);
      CHECK(H(1, 1) == 200);
    }
  }
}

TEST_CASE("Hessian - Reuse", "[Hessian]") {
  sleipnir::Variable y;
  sleipnir::VariableMatrix x{1};

  // y = x³
  x(0).SetValue(1);
  y = x(0) * x(0) * x(0);

  sleipnir::Hessian hessian{y, x};

  // d²y/dx² = 6x
  // H = 6
  Eigen::MatrixXd H = hessian.Value();

  CHECK(H.rows() == 1);
  CHECK(H.cols() == 1);
  CHECK(H(0, 0) == 6.0);

  x(0).SetValue(2);
  // d²y/dx² = 6x
  // H = 12
  H = hessian.Value();

  CHECK(H.rows() == 1);
  CHECK(H.cols() == 1);
  CHECK(H(0, 0) == 12.0);
}
