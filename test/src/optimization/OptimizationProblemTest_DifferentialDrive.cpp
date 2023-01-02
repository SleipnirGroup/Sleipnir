// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <fstream>
#include <numbers>

#include <Eigen/Core>
#include <fmt/core.h>
#include <gtest/gtest.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <units/acceleration.h>
#include <units/angle.h>
#include <units/length.h>
#include <units/time.h>
#include <units/velocity.h>
#include <units/voltage.h>

#include "RK4.hpp"

Eigen::Vector<double, 5> DifferentialDriveDynamicsDouble(
    const Eigen::Vector<double, 5>& x, const Eigen::Vector<double, 2>& u) {
  // x = [x, y, heading, left velocity, right velocity]ᵀ
  // u = [left voltage, right voltage]ᵀ
  constexpr double trackwidth = (0.699_m).value();
  constexpr double Kv_linear = (3.02_V / 1_mps).value();
  constexpr double Ka_linear = (0.642_V / 1_mps_sq).value();
  constexpr double Kv_angular = (1.382_V / 1_mps).value();
  constexpr double Ka_angular = (0.08495_V / 1_mps_sq).value();

  double v = (x(3) + x(4)) / 2.0;

  constexpr double A1 =
      -(Kv_linear / Ka_linear + Kv_angular / Ka_angular) / 2.0;
  constexpr double A2 =
      -(Kv_linear / Ka_linear - Kv_angular / Ka_angular) / 2.0;
  constexpr double B1 = 0.5 / Ka_linear + 0.5 / Ka_angular;
  constexpr double B2 = 0.5 / Ka_linear - 0.5 / Ka_angular;
  Eigen::Matrix<double, 2, 2> A{{A1, A2}, {A2, A1}};
  Eigen::Matrix<double, 2, 2> B{{B1, B2}, {B2, B1}};

  Eigen::Vector<double, 5> xdot;
  xdot(0) = v * std::cos(x(2));
  xdot(1) = v * std::sin(x(2));
  xdot(2) = (x(4) - x(3)) / trackwidth;
  xdot.block<2, 1>(3, 0) = A * x.block<2, 1>(3, 0) + B * u;
  return xdot;
}

sleipnir::VariableMatrix DifferentialDriveDynamics(
    const sleipnir::VariableMatrix& x, const sleipnir::VariableMatrix& u) {
  // x = [x, y, heading, left velocity, right velocity]ᵀ
  // u = [left voltage, right voltage]ᵀ
  constexpr double trackwidth = (0.699_m).value();
  constexpr double Kv_linear = (3.02_V / 1_mps).value();
  constexpr double Ka_linear = (0.642_V / 1_mps_sq).value();
  constexpr double Kv_angular = (1.382_V / 1_mps).value();
  constexpr double Ka_angular = (0.08495_V / 1_mps_sq).value();

  auto v = (x(3) + x(4)) / 2.0;

  constexpr double A1 =
      -(Kv_linear / Ka_linear + Kv_angular / Ka_angular) / 2.0;
  constexpr double A2 =
      -(Kv_linear / Ka_linear - Kv_angular / Ka_angular) / 2.0;
  constexpr double B1 = 0.5 / Ka_linear + 0.5 / Ka_angular;
  constexpr double B2 = 0.5 / Ka_linear - 0.5 / Ka_angular;
  Eigen::Matrix<double, 2, 2> A{{A1, A2}, {A2, A1}};
  Eigen::Matrix<double, 2, 2> B{{B1, B2}, {B2, B1}};

  sleipnir::VariableMatrix xdot{5, 1};
  xdot(0) = v * sleipnir::cos(x(2));
  xdot(1) = v * sleipnir::sin(x(2));
  xdot(2) = (x(4) - x(3)) / trackwidth;
  xdot.Segment(3, 2) = A * x.Segment(3, 2) + B * u;
  return xdot;
}

TEST(OptimizationProblemTest, DifferentialDrive) {
  constexpr auto T = 5_s;
  constexpr units::second_t dt = 50_ms;
  constexpr int N = T / dt;

  constexpr auto u_max = 12_V;

  constexpr Eigen::Vector<double, 5> x_initial{{0.0, 0.0, 0.0, 0.0, 0.0}};
  constexpr Eigen::Vector<double, 5> x_final{{1.0, 1.0, 0.0, 0.0, 0.0}};

  auto start = std::chrono::system_clock::now();

  sleipnir::OptimizationProblem problem;

  // x = [x, y, heading, left velocity, right velocity]ᵀ
  auto X = problem.DecisionVariable(5, N + 1);

  // Initial guess
  for (int k = 0; k < N; ++k) {
    X(0, k).SetValue(
        std::lerp(x_initial(0), x_final(0), static_cast<double>(k) / N));
    X(1, k).SetValue(
        std::lerp(x_initial(1), x_final(1), static_cast<double>(k) / N));
  }

  // u = [left voltage, right voltage]ᵀ
  auto U = problem.DecisionVariable(2, N);

  // Initial conditions
  problem.SubjectTo(X.Col(0) == x_initial);

  // Final conditions
  problem.SubjectTo(X.Col(N) == x_final);

  // Input constraints
  problem.SubjectTo(U >= -u_max.value());
  problem.SubjectTo(U <= u_max.value());

  // Dynamics constraints - RK4 integration
  for (int k = 0; k < N; ++k) {
    problem.SubjectTo(X.Col(k + 1) ==
                      RK4<decltype(DifferentialDriveDynamics),
                          sleipnir::VariableMatrix, sleipnir::VariableMatrix>(
                          DifferentialDriveDynamics, X.Col(k), U.Col(k), dt));
  }

  // Minimize sum squared states and inputs
  sleipnir::Variable J = 0.0;
  for (int k = 0; k < N; ++k) {
    J += X.Col(k).T() * X.Col(k) + U.Col(k).T() * U.Col(k);
  }
  problem.Minimize(J);

  auto end1 = std::chrono::system_clock::now();
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  fmt::print("Setup time: {} ms\n\n",
             duration_cast<microseconds>(end1 - start).count() / 1000.0);

  auto status = problem.Solve({.diagnostics = true});

  EXPECT_EQ(sleipnir::ExpressionType::kQuadratic, status.costFunctionType);
  EXPECT_EQ(sleipnir::ExpressionType::kNonlinear,
            status.equalityConstraintType);
  EXPECT_EQ(sleipnir::ExpressionType::kLinear, status.inequalityConstraintType);
  EXPECT_EQ(sleipnir::SolverExitCondition::kSuccess, status.exitCondition);

  // Verify initial state
  EXPECT_NEAR(x_initial(0), X.Value(0, 0), 1e-2);
  EXPECT_NEAR(x_initial(1), X.Value(1, 0), 1e-2);
  EXPECT_NEAR(x_initial(2), X.Value(2, 0), 1e-2);
  EXPECT_NEAR(x_initial(3), X.Value(3, 0), 1e-2);
  EXPECT_NEAR(x_initial(4), X.Value(4, 0), 1e-2);

  // Verify solution
  Eigen::Vector<double, 5> x{0.0, 0.0, 0.0, 0.0, 0.0};
  Eigen::Vector<double, 2> u{0.0, 0.0};
  for (int k = 0; k < N; ++k) {
    u = U.Col(k).Value();

    // Input constraints
    EXPECT_GE(U(0, k).Value(), -u_max.value());
    EXPECT_LE(U(0, k).Value(), u_max.value());
    EXPECT_GE(U(1, k).Value(), -u_max.value());
    EXPECT_LE(U(1, k).Value(), u_max.value());

    // Verify state
    EXPECT_NEAR(x(0), X.Value(0, k), 1e-2) << fmt::format("  k = {}", k);
    EXPECT_NEAR(x(1), X.Value(1, k), 1e-2) << fmt::format("  k = {}", k);
    EXPECT_NEAR(x(2), X.Value(2, k), 1e-2) << fmt::format("  k = {}", k);
    EXPECT_NEAR(x(3), X.Value(3, k), 1e-2) << fmt::format("  k = {}", k);
    EXPECT_NEAR(x(4), X.Value(4, k), 1e-2) << fmt::format("  k = {}", k);

    // Project state forward
    x = RK4(DifferentialDriveDynamicsDouble, x, u, dt);
  }

  // Verify final state
  EXPECT_NEAR(x_final(0), X.Value(0, N), 1e-2);
  EXPECT_NEAR(x_final(1), X.Value(1, N), 1e-2);
  EXPECT_NEAR(x_final(2), X.Value(2, N), 1e-2);
  EXPECT_NEAR(x_final(3), X.Value(3, N), 1e-2);
  EXPECT_NEAR(x_final(4), X.Value(4, N), 1e-2);

  // Log states for offline viewing
  std::ofstream states{"Differential drive states.csv"};
  if (states.is_open()) {
    states << "Time (s),X position (m),Y position (m),Heading (rad),Left "
              "velocity (m/s),Right velocity (m/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << fmt::format("{},{},{},{},{},{}\n", k * dt.value(),
                            X.Value(0, k), X.Value(1, k), X.Value(2, k),
                            X.Value(3, k), X.Value(4, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"Differential drive inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Left voltage (V),Right voltage (V)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << fmt::format("{},{},{}\n", k * dt.value(), U.Value(0, k),
                              U.Value(1, k));
      } else {
        inputs << fmt::format("{},{},{}\n", k * dt.value(), 0.0, 0.0);
      }
    }
  }
}
