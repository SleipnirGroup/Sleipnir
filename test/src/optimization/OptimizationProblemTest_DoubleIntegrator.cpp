// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <fstream>

#include <Eigen/Core>
#include <fmt/core.h>
#include <gtest/gtest.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <units/time.h>

#include "CmdlineArguments.hpp"

TEST(OptimizationProblemTest, DoubleIntegrator) {
  auto start = std::chrono::system_clock::now();

  constexpr auto T = 3.5_s;
  constexpr units::second_t dt = 5_ms;
  constexpr int N = T / dt;

  constexpr double r = 2.0;

  sleipnir::OptimizationProblem problem;

  // 2x1 state vector with N + 1 timesteps (includes last state)
  auto X = problem.DecisionVariable(2, N + 1);

  // 1x1 input vector with N timesteps (input at last state doesn't matter)
  auto U = problem.DecisionVariable(1, N);

  // Kinematics constraint assuming constant acceleration between timesteps
  for (int k = 0; k < N; ++k) {
    constexpr double t = dt.value();
    auto p_k1 = X(0, k + 1);
    auto v_k1 = X(1, k + 1);
    auto p_k = X(0, k);
    auto v_k = X(1, k);
    auto a_k = U(0, k);

    // pₖ₊₁ = pₖ + vₖt
    problem.SubjectTo(p_k1 == p_k + v_k * t);

    // vₖ₊₁ = vₖ + aₖt
    problem.SubjectTo(v_k1 == v_k + a_k * t);
  }

  // Start and end at rest
  problem.SubjectTo(X.Col(0) == Eigen::Matrix<double, 2, 1>{{0.0}, {0.0}});
  problem.SubjectTo(X.Col(N) == Eigen::Matrix<double, 2, 1>{{r}, {0.0}});

  // Limit velocity
  problem.SubjectTo(-1 <= X.Row(1));
  problem.SubjectTo(X.Row(1) <= 1);

  // Limit acceleration
  problem.SubjectTo(-1 <= U);
  problem.SubjectTo(U <= 1);

  // Cost function - minimize position error
  sleipnir::Variable J = 0.0;
  for (int k = 0; k < N + 1; ++k) {
    J += sleipnir::pow(r - X(0, k), 2);
  }
  problem.Minimize(J);

  [[maybe_unused]] auto end1 = std::chrono::system_clock::now();
  if (CmdlineArgPresent(kEnableDiagnostics)) {
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    fmt::print("Setup time: {} ms\n\n",
               duration_cast<microseconds>(end1 - start).count() / 1000.0);
  }

  auto status =
      problem.Solve({.diagnostics = CmdlineArgPresent(kEnableDiagnostics)});

  EXPECT_EQ(sleipnir::ExpressionType::kQuadratic, status.costFunctionType);
  EXPECT_EQ(sleipnir::ExpressionType::kLinear, status.equalityConstraintType);
  EXPECT_EQ(sleipnir::ExpressionType::kLinear, status.inequalityConstraintType);
  EXPECT_EQ(sleipnir::SolverExitCondition::kSuccess, status.exitCondition);

  Eigen::Matrix<double, 2, 2> A{{1.0, dt.value()}, {0.0, 1.0}};
  Eigen::Matrix<double, 2, 1> B{0.5 * dt.value() * dt.value(), dt.value()};

  // Verify solution
  Eigen::Matrix<double, 2, 1> x{0.0, 0.0};
  Eigen::Matrix<double, 1, 1> u{0.0};
  for (int k = 0; k < N; ++k) {
    // Verify state
    EXPECT_NEAR(x(0), X.Value(0, k), 1e-2) << fmt::format("  k = {}", k);
    EXPECT_NEAR(x(1), X.Value(1, k), 1e-2) << fmt::format("  k = {}", k);

    // Determine expected input for this timestep
    if (k * dt < 1_s) {
      // Accelerate
      u(0) = 1.0;
    } else if (k * dt < 2.05_s) {
      // Maintain speed
      u(0) = 0.0;
    } else if (k * dt < 3.275_s) {
      // Decelerate
      u(0) = -1.0;
    } else {
      // Accelerate
      u(0) = 1.0;
    }

    // Verify input
    if (k > 0 && k < N - 1 &&
        std::abs(U.Value(0, k - 1) - U.Value(0, k + 1)) >= 1.0 - 1e-2) {
      // If control input is transitioning between -1, 0, or 1, ensure it's
      // within (-1, 1)
      EXPECT_GE(u(0), -1.0) << fmt::format("  k = {}", k);
      EXPECT_LE(u(0), 1.0) << fmt::format("  k = {}", k);
    } else {
      EXPECT_NEAR(u(0), U.Value(0, k), 1e-2) << fmt::format("  k = {}", k);
    }

    // Project state forward
    x = A * x + B * u;
  }

  // Verify final state
  EXPECT_NEAR(r, X.Value(0, N), 1e-2);
  EXPECT_NEAR(0.0, X.Value(1, N), 1e-2);

  // Log states for offline viewing
  std::ofstream states{"OptimizationProblem Double Integrator states.csv"};
  if (states.is_open()) {
    states << "Time (s),Position (m),Velocity (m/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << fmt::format("{},{},{}\n", k * dt.value(), X.Value(0, k),
                            X.Value(1, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"OptimizationProblem Double Integrator inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Acceleration (m/s²)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << fmt::format("{},{}\n", k * dt.value(), U.Value(0, k));
      } else {
        inputs << fmt::format("{},{}\n", k * dt.value(), 0.0);
      }
    }
  }
}
