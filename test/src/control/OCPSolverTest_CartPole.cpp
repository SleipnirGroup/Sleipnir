// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <fstream>
#include <numbers>

#include <Eigen/Core>
#include <fmt/core.h>
#include <gtest/gtest.h>
#include <sleipnir/control/OCPSolver.hpp>
#include <units/acceleration.h>
#include <units/force.h>
#include <units/length.h>
#include <units/time.h>

#include "CartPoleUtil.hpp"
#include "CmdlineArguments.hpp"

TEST(OCPSolverTest, CartPole) {
  constexpr auto T = 5_s;
  constexpr units::second_t dt = 50_ms;
  constexpr int N = T / dt;

  constexpr auto u_max = 20_N;
  constexpr auto d = 1_m;
  constexpr auto d_max = 2_m;

  auto start = std::chrono::system_clock::now();

  auto dynamicsFunction = [=](sleipnir::Variable t, sleipnir::VariableMatrix x,
                              sleipnir::VariableMatrix u,
                              sleipnir::Variable dt) {
    return CartPoleDynamics(x, u);
  };

  sleipnir::OCPSolver problem(
      4, 1, std::chrono::duration<double>{dt.value()}, N, dynamicsFunction,
      sleipnir::DynamicsType::kExplicitODE,
      sleipnir::TimestepMethod::kVariableSingle,
      sleipnir::TranscriptionMethod::kDirectCollocation);

  // x = [q, q̇]ᵀ = [x, θ, ẋ, θ̇]ᵀ
  auto X = problem.X();

  // Initial guess
  for (int k = 0; k < N + 1; ++k) {
    X(0, k).SetValue(static_cast<double>(k) / N * d.value());
    X(1, k).SetValue(static_cast<double>(k) / N * std::numbers::pi);
  }

  // Initial conditions
  problem.ConstrainInitialState(
      Eigen::Matrix<double, 4, 1>{0.0, 0.0, 0.0, 0.0});

  // Final conditions
  problem.ConstrainFinalState(
      Eigen::Matrix<double, 4, 1>{1.0, std::numbers::pi, 0.0, 0.0});

  // Cart position constraints
  problem.SubjectTo(X.Row(0) >= 0.0);
  problem.SubjectTo(X.Row(0) <= d_max.value());

  // Input constraints
  problem.SetLowerInputBound(-u_max.value());
  problem.SetUpperInputBound(u_max.value());

  // Minimize sum squared inputs
  sleipnir::Variable J = 0.0;
  for (int k = 0; k < N; ++k) {
    J += problem.U().Col(k).T() * problem.U().Col(k);
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
  // FIXME: Fails with "bad search direction"
  // EXPECT_EQ(sleipnir::SolverExitCondition::kSuccess, status.exitCondition);

#if 0
  // Verify initial state
  EXPECT_NEAR(0.0, X.Value(0, 0), 1e-2);
  EXPECT_NEAR(0.0, X.Value(1, 0), 1e-2);
  EXPECT_NEAR(0.0, X.Value(2, 0), 1e-2);
  EXPECT_NEAR(0.0, X.Value(3, 0), 1e-2);

  // Verify solution
  Eigen::Matrix<double, 4, 1> x{0.0, 0.0, 0.0, 0.0};
  Eigen::Matrix<double, 1, 1> u{0.0};
  for (int k = 0; k < N; ++k) {
    u = problem.U().Col(k).Value();

    // Verify state
    EXPECT_NEAR(x(0), X.Value(0, k), 1e-2) << fmt::format("  k = {}", k);
    EXPECT_NEAR(x(1), X.Value(1, k), 1e-2) << fmt::format("  k = {}", k);
    EXPECT_NEAR(x(2), X.Value(2, k), 1e-2) << fmt::format("  k = {}", k);
    EXPECT_NEAR(x(3), X.Value(3, k), 1e-2) << fmt::format("  k = {}", k);

    // Project state forward
    x = RK4(CartPoleDynamicsDouble, x, u, dt);
  }

  // Verify final state
  EXPECT_NEAR(1.0, X.Value(0, N - 1), 1e-2);
  EXPECT_NEAR(std::numbers::pi, X.Value(1, N - 1), 1e-2);
  EXPECT_NEAR(0.0, X.Value(2, N - 1), 1e-2);
  EXPECT_NEAR(0.0, X.Value(3, N - 1), 1e-2);
#endif

  // Log states for offline viewing
  std::ofstream states{"OCPSolver Cart-pole states.csv"};
  if (states.is_open()) {
    states << "Time (s),Cart position (m),Pole angle (rad),Cart velocity (m/s),"
              "Pole angular velocity (rad/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << fmt::format("{},{},{},{},{}\n", k * dt.value(), X.Value(0, k),
                            X.Value(1, k), X.Value(2, k), X.Value(3, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"OCPSolver Cart-pole inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Cart force (N)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << fmt::format("{},{}\n", k * dt.value(),
                              problem.U().Value(0, k));
      } else {
        inputs << fmt::format("{},{}\n", k * dt.value(), 0.0);
      }
    }
  }
}
