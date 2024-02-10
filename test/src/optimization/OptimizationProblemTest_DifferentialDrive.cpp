// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <fstream>
#include <numbers>

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/core.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <units/time.h>
#include <units/voltage.h>

#include "CmdlineArguments.hpp"
#include "DifferentialDriveUtil.hpp"
#include "RK4.hpp"

TEST_CASE("Differential drive", "[OptimizationProblem]") {
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

  [[maybe_unused]] auto end1 = std::chrono::system_clock::now();
  if (Argv().Contains("--enable-diagnostics")) {
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    fmt::print("Setup time: {} ms\n\n",
               duration_cast<microseconds>(end1 - start).count() / 1000.0);
  }

  auto status =
      problem.Solve({.diagnostics = Argv().Contains("--enable-diagnostics")});

  CHECK(status.costFunctionType == sleipnir::ExpressionType::kQuadratic);
  CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kNonlinear);
  CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kLinear);
  CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);

  // Verify initial state
  CHECK(X.Value(0, 0) == Catch::Approx(x_initial(0)).margin(1e-8));
  CHECK(X.Value(1, 0) == Catch::Approx(x_initial(1)).margin(1e-8));
  CHECK(X.Value(2, 0) == Catch::Approx(x_initial(2)).margin(1e-8));
  CHECK(X.Value(3, 0) == Catch::Approx(x_initial(3)).margin(1e-8));
  CHECK(X.Value(4, 0) == Catch::Approx(x_initial(4)).margin(1e-8));

  // Verify solution
  Eigen::Vector<double, 5> x{0.0, 0.0, 0.0, 0.0, 0.0};
  Eigen::Vector<double, 2> u{0.0, 0.0};
  for (int k = 0; k < N; ++k) {
    u = U.Col(k).Value();

    // Input constraints
    CHECK(U(0, k).Value() >= -u_max.value());
    CHECK(U(0, k).Value() <= u_max.value());
    CHECK(U(1, k).Value() >= -u_max.value());
    CHECK(U(1, k).Value() <= u_max.value());

    // Verify state
    CHECK(X.Value(0, k) == Catch::Approx(x(0)).margin(1e-8));
    CHECK(X.Value(1, k) == Catch::Approx(x(1)).margin(1e-8));
    CHECK(X.Value(2, k) == Catch::Approx(x(2)).margin(1e-8));
    CHECK(X.Value(3, k) == Catch::Approx(x(3)).margin(1e-8));
    CHECK(X.Value(4, k) == Catch::Approx(x(4)).margin(1e-8));
    INFO(fmt::format("  k = {}", k));

    // Project state forward
    x = RK4(DifferentialDriveDynamicsDouble, x, u, dt);
  }

  // Verify final state
  CHECK(X.Value(0, N) == Catch::Approx(x_final(0)).margin(1e-8));
  CHECK(X.Value(1, N) == Catch::Approx(x_final(1)).margin(1e-8));
  CHECK(X.Value(2, N) == Catch::Approx(x_final(2)).margin(1e-8));
  CHECK(X.Value(3, N) == Catch::Approx(x_final(3)).margin(1e-8));
  CHECK(X.Value(4, N) == Catch::Approx(x_final(4)).margin(1e-8));

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
