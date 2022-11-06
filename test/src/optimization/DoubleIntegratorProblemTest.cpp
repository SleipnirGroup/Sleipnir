// Copyright (c) Joshua Nichols and Tyler Veness

#include <chrono>
#include <fstream>

#include <Eigen/Core>
#include <fmt/core.h>
#include <gtest/gtest.h>
#include <sleipnir/optimization/OptimizationProblem.h>
#include <units/time.h>

TEST(DoubleIntegratorProblemTest, MinimumTime) {
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
  sleipnir::VariableMatrix J = 0.0;
  for (int k = 0; k < N + 1; ++k) {
    J += sleipnir::pow(r - X(0, k), 2);
  }
  problem.Minimize(J);

  auto end1 = std::chrono::system_clock::now();
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  fmt::print("Setup time: {} ms\n\n",
             duration_cast<microseconds>(end1 - start).count() / 1000.0);

  sleipnir::SolverConfig config;
  config.diagnostics = true;

  auto status = problem.Solve(config);

  EXPECT_EQ(sleipnir::autodiff::ExpressionType::kQuadratic,
            status.costFunctionType);
  EXPECT_EQ(sleipnir::autodiff::ExpressionType::kLinear,
            status.equalityConstraintType);
  EXPECT_EQ(sleipnir::autodiff::ExpressionType::kLinear,
            status.inequalityConstraintType);
  EXPECT_EQ(sleipnir::SolverExitCondition::kOk, status.exitCondition);

  // TODO: Verify solution

  // Log states for offline viewing
  std::ofstream states{"Double integrator states.csv"};
  if (states.is_open()) {
    states << "Time (s),Position (m),Velocity (m/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << fmt::format("{},{},{}\n", k * dt.value(), X.Value(0, k),
                            X.Value(1, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"Double integrator inputs.csv"};
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
