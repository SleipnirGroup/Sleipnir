// Copyright (c) Sleipnir contributors

#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/Jacobian.hpp>

TEST_CASE("Jacobian - y = x", "[Jacobian]") {
  sleipnir::VariableMatrix y{3};
  sleipnir::VariableMatrix x{3};
  x(0).SetValue(1);
  x(1).SetValue(2);
  x(2).SetValue(3);

  // y = x
  //
  //         [1  0  0]
  // dy/dx = [0  1  0]
  //         [0  0  1]
  y = x;
  Eigen::MatrixXd J = sleipnir::Jacobian(y, x).Value();

  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      if (row == col) {
        CHECK(J(row, col) == 1.0);
      } else {
        CHECK(J(row, col) == 0.0);
      }
    }
  }
}

TEST_CASE("Jacobian - y = 3x", "[Jacobian]") {
  sleipnir::VariableMatrix y{3};
  sleipnir::VariableMatrix x{3};
  x(0).SetValue(1);
  x(1).SetValue(2);
  x(2).SetValue(3);

  // y = 3x
  //
  //         [3  0  0]
  // dy/dx = [0  3  0]
  //         [0  0  3]
  y = 3 * x;
  Eigen::MatrixXd J = sleipnir::Jacobian(y, x).Value();

  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      if (row == col) {
        CHECK(J(row, col) == 3.0);
      } else {
        CHECK(J(row, col) == 0.0);
      }
    }
  }
}

TEST_CASE("Jacobian - Products", "[Jacobian]") {
  sleipnir::VariableMatrix y{3};
  sleipnir::VariableMatrix x{3};
  x(0).SetValue(1);
  x(1).SetValue(2);
  x(2).SetValue(3);

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
  Eigen::MatrixXd J = sleipnir::Jacobian(y, x).Value();

  Eigen::MatrixXd expectedJ{{2.0, 1.0, 0.0}, {0.0, 3.0, 2.0}, {3.0, 0.0, 1.0}};
  for (int i = 0; i < x.Rows(); ++i) {
    for (int j = 0; j < x.Rows(); ++j) {
      CHECK(J(i, j) == expectedJ(i, j));
    }
  }
}

TEST_CASE("Jacobian - Nested products", "[Jacobian]") {
  SKIP("Fails");

  sleipnir::VariableMatrix z{1};
  z(0).SetValue(1);
  sleipnir::VariableMatrix x{3};
  x(0) = 1 * z(0);
  x(1) = 2 * z(0);
  x(2) = 3 * z(0);

  Eigen::MatrixXd J = sleipnir::Jacobian(x, z).Value();
  CHECK(J(0, 0) == 1.0);
  CHECK(J(1, 0) == 2.0);
  CHECK(J(2, 0) == 3.0);

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
  sleipnir::VariableMatrix y{3};
  y(0) = x(0) * x(1);
  y(1) = x(1) * x(2);
  y(2) = x(0) * x(2);
  J = sleipnir::Jacobian(y, x).Value();

  Eigen::MatrixXd expectedJ{{2.0, 1.0, 0.0}, {0.0, 3.0, 2.0}, {3.0, 0.0, 1.0}};
  for (int i = 0; i < x.Rows(); ++i) {
    for (int j = 0; j < x.Rows(); ++j) {
      CHECK(J(i, j) == expectedJ(i, j));
    }
  }
}

TEST_CASE("Jacobian - Non-square", "[Jacobian]") {
  sleipnir::VariableMatrix y{1};
  sleipnir::VariableMatrix x{3};
  x(0).SetValue(1);
  x(1).SetValue(2);
  x(2).SetValue(3);

  // y = [x₁ + 3x₂ − 5x₃]
  //
  // dy/dx = [1  3  −5]
  y(0) = x(0) + 3 * x(1) - 5 * x(2);
  Eigen::MatrixXd J = sleipnir::Jacobian(y, x).Value();

  CHECK(J.rows() == 1);
  CHECK(J.cols() == 3);
  CHECK(J(0, 0) == 1.0);
  CHECK(J(0, 1) == 3.0);
  CHECK(J(0, 2) == -5.0);
}

TEST_CASE("Jacobian - Reuse", "[Jacobian]") {
  sleipnir::VariableMatrix y{1};
  sleipnir::VariableMatrix x{2};

  // y = [x₁x₂]
  x(0).SetValue(1);
  x(1).SetValue(2);
  y(0) = x(0) * x(1);

  sleipnir::Jacobian jacobian{y, x};

  // dy/dx = [x₂  x₁]
  // dy/dx = [2  1]
  Eigen::MatrixXd J = jacobian.Value();

  CHECK(J.rows() == 1);
  CHECK(J.cols() == 2);
  CHECK(J(0, 0) == 2.0);
  CHECK(J(0, 1) == 1.0);

  x(0).SetValue(2);
  x(1).SetValue(1);
  // dy/dx = [x₂  x₁]
  // dy/dx = [1  2]
  J = jacobian.Value();

  CHECK(J.rows() == 1);
  CHECK(J.cols() == 2);
  CHECK(J(0, 0) == 1.0);
  CHECK(J(0, 1) == 2.0);
}
