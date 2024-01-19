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
#include <units/force.h>
#include <units/length.h>
#include <units/time.h>

#include "CartPoleUtil.hpp"
#include "CmdlineArguments.hpp"
#include "RK4.hpp"

TEST(OptimizationProblemTest, CartPole) {
  constexpr auto T = 5_s;
  constexpr units::second_t dt = 50_ms;
  constexpr int N = T / dt;

  constexpr auto u_max = 20_N;
  constexpr auto d_max = 2_m;

  constexpr Eigen::Vector<double, 4> x_initial{{0.0, 0.0, 0.0, 0.0}};
  constexpr Eigen::Vector<double, 4> x_final{{1.0, std::numbers::pi, 0.0, 0.0}};

  auto start = std::chrono::system_clock::now();

  sleipnir::OptimizationProblem problem;

  // x = [q, q̇]ᵀ = [x, θ, ẋ, θ̇]ᵀ
  auto X = problem.DecisionVariable(4, N + 1);

  // Initial guess
  for (int k = 0; k < N + 1; ++k) {
    X(0, k).SetValue(
        std::lerp(x_initial(0), x_final(0), static_cast<double>(k) / N));
    X(1, k).SetValue(
        std::lerp(x_initial(1), x_final(1), static_cast<double>(k) / N));
  }

  // u = f_x
  auto U = problem.DecisionVariable(1, N);

  // Initial conditions
  problem.SubjectTo(X.Col(0) == x_initial);

  // Final conditions
  problem.SubjectTo(X.Col(N) == x_final);

  // Cart position constraints
  problem.SubjectTo(X.Row(0) >= 0.0);
  problem.SubjectTo(X.Row(0) <= d_max.value());

  // Input constraints
  problem.SubjectTo(U >= -u_max.value());
  problem.SubjectTo(U <= u_max.value());

  // Dynamics constraints - RK4 integration
  for (int k = 0; k < N; ++k) {
    problem.SubjectTo(X.Col(k + 1) ==
                      RK4<decltype(CartPoleDynamics), sleipnir::VariableMatrix,
                          sleipnir::VariableMatrix>(CartPoleDynamics, X.Col(k),
                                                    U.Col(k), dt));
  }

  // Minimize sum squared inputs
  sleipnir::Variable J = 0.0;
  for (int k = 0; k < N; ++k) {
    J += U.Col(k).T() * U.Col(k);
  }
  problem.Minimize(J);

  [[maybe_unused]] auto end1 = std::chrono::system_clock::now();
  if (Argv().Contains("--enable-diagnostics")) {
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    fmt::print("Setup time: {} ms\n\n",
               duration_cast<microseconds>(end1 - start).count() / 1000.0);
  }

  auto status =
      problem.Solve({.diagnostics = Argv().Contains("--enable-diagnostics")});

  EXPECT_EQ(sleipnir::ExpressionType::kQuadratic, status.costFunctionType);
  EXPECT_EQ(sleipnir::ExpressionType::kNonlinear,
            status.equalityConstraintType);
  EXPECT_EQ(sleipnir::ExpressionType::kLinear, status.inequalityConstraintType);
  EXPECT_EQ(sleipnir::SolverExitCondition::kSuccess, status.exitCondition);

  // Verify initial state
  EXPECT_NEAR(x_initial(0), X.Value(0, 0), 1e-8);
  EXPECT_NEAR(x_initial(1), X.Value(1, 0), 1e-8);
  EXPECT_NEAR(x_initial(2), X.Value(2, 0), 1e-8);
  EXPECT_NEAR(x_initial(3), X.Value(3, 0), 1e-8);

  // Verify solution
  for (int k = 0; k < N; ++k) {
    // Cart position constraints
    EXPECT_GE(X(0, k), 0.0);
    EXPECT_LE(X(0, k), d_max.value());

    // Input constraints
    EXPECT_GE(U(0, k), -u_max.value());
    EXPECT_LE(U(0, k), u_max.value());

    // Dynamics constraints
    Eigen::VectorXd expected_x_k1 =
        RK4(CartPoleDynamicsDouble, X.Col(k).Value(), U.Col(k).Value(), dt);
    Eigen::VectorXd actual_x_k1 = X.Col(k + 1).Value();
    for (int row = 0; row < actual_x_k1.rows(); ++row) {
      EXPECT_NEAR(expected_x_k1(row), actual_x_k1(row), 1e-8)
          << "  x(" << row << ") @ k = " << k;
    }
  }

  // Verify final state
  EXPECT_NEAR(x_final(0), X.Value(0, N), 1e-8);
  EXPECT_NEAR(x_final(1), X.Value(1, N), 1e-8);
  EXPECT_NEAR(x_final(2), X.Value(2, N), 1e-8);
  EXPECT_NEAR(x_final(3), X.Value(3, N), 1e-8);

  // Log states for offline viewing
  std::ofstream states{"OptimizationProblem Cart-pole states.csv"};
  if (states.is_open()) {
    states << "Time (s),Cart position (m),Pole angle (rad),Cart velocity (m/s),"
              "Pole angular velocity (rad/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << fmt::format("{},{},{},{},{}\n", k * dt.value(), X.Value(0, k),
                            X.Value(1, k), X.Value(2, k), X.Value(3, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"OptimizationProblem Cart-pole inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Cart force (N)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << fmt::format("{},{}\n", k * dt.value(), U.Value(0, k));
      } else {
        inputs << fmt::format("{},{}\n", k * dt.value(), 0.0);
      }
    }
  }
}
