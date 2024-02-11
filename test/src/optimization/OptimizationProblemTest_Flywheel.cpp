// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/QR>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/core.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <units/angle.h>
#include <units/angular_acceleration.h>
#include <units/angular_velocity.h>
#include <units/time.h>
#include <units/voltage.h>

#include "CmdlineArguments.hpp"

namespace {
bool Near(double expected, double actual, double tolerance) {
  return std::abs(expected - actual) < tolerance;
}
}  // namespace

TEST_CASE("Flywheel", "[OptimizationProblem]") {
  auto start = std::chrono::system_clock::now();

  constexpr auto T = 5_s;
  constexpr units::second_t dt = 5_ms;
  constexpr int N = T / dt;

  // Flywheel model:
  // States: [velocity]
  // Inputs: [voltage]
  Eigen::Matrix<double, 1, 1> A{std::exp(-dt.value())};
  Eigen::Matrix<double, 1, 1> B{1.0 - std::exp(-dt.value())};

  sleipnir::OptimizationProblem problem;
  auto X = problem.DecisionVariable(1, N + 1);
  auto U = problem.DecisionVariable(1, N);

  // Dynamics constraint
  for (int k = 0; k < N; ++k) {
    problem.SubjectTo(X.Col(k + 1) == A * X.Col(k) + B * U.Col(k));
  }

  // State and input constraints
  problem.SubjectTo(X.Col(0) == 0.0);
  problem.SubjectTo(-12 <= U);
  problem.SubjectTo(U <= 12);

  // Cost function - minimize error
  Eigen::Matrix<double, 1, 1> r{10.0};
  sleipnir::Variable J = 0.0;
  for (int k = 0; k < N + 1; ++k) {
    J += (r - X.Col(k)).T() * (r - X.Col(k));
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
  CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kLinear);
  CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kLinear);
  CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);

  // Voltage for steady-state velocity:
  //
  // rₖ₊₁ = Arₖ + Buₖ
  // uₖ = B⁺(rₖ₊₁ − Arₖ)
  // uₖ = B⁺(rₖ − Arₖ)
  // uₖ = B⁺(I − A)rₖ
  Eigen::Matrix<double, 1, 1> u_ss =
      B.householderQr().solve(decltype(A)::Identity() - A) * r;

  // Verify initial state
  CHECK(X.Value(0, 0) == Catch::Approx(0.0).margin(1e-8));

  // Verify solution
  Eigen::Matrix<double, 1, 1> x{0.0};
  Eigen::Matrix<double, 1, 1> u{0.0};
  for (int k = 0; k < N; ++k) {
    // Verify state
    CHECK(X.Value(0, k) == Catch::Approx(x(0)).margin(1e-2));

    // Determine expected input for this timestep
    double error = r(0) - x(0);
    if (error > 1e-2) {
      // Max control input until the reference is reached
      u(0) = 12.0;
    } else {
      // Maintain speed
      u = u_ss;
    }

    // Verify input
    if (k > 0 && k < N - 1 && Near(12.0, U.Value(0, k - 1), 1e-2) &&
        Near(u_ss(0), U.Value(0, k + 1), 1e-2)) {
      // If control input is transitioning between 12 and u_ss, ensure it's
      // within (u_ss, 12)
      CHECK(U.Value(0, k) >= u_ss(0));
      CHECK(U.Value(0, k) <= 12.0);
    } else {
      CHECK(U.Value(0, k) == Catch::Approx(u(0)).margin(1e-4));
    }

    INFO(fmt::format("  k = {}", k));

    // Project state forward
    x = A * x + B * u;
  }

  // Verify final state
  CHECK(X.Value(0, N) == Catch::Approx(r(0)).margin(1e-7));

  // Log states for offline viewing
  std::ofstream states{"OptimizationProblem Flywheel states.csv"};
  if (states.is_open()) {
    states << "Time (s),Velocity (rad/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << fmt::format("{},{}\n", k * units::second_t{dt}.value(),
                            X.Value(0, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"OptimizationProblem Flywheel inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Voltage (V)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << fmt::format("{},{}\n", k * dt.value(), U.Value(0, k));
      } else {
        inputs << fmt::format("{},{}\n", k * dt.value(), 0.0);
      }
    }
  }
}
