// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <fstream>
#include <numbers>

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/core.h>
#include <sleipnir/control/OCPSolver.hpp>
#include <units/acceleration.h>
#include <units/force.h>
#include <units/length.h>
#include <units/time.h>

#include "CartPoleUtil.hpp"
#include "RK4.hpp"

TEST_CASE("OCPSolver - Cart-pole", "[OCPSolver]") {
  constexpr auto T = 5_s;
  constexpr units::second_t dt = 50_ms;
  constexpr int N = T / dt;

  constexpr auto u_max = 20_N;
  constexpr auto d_max = 2_m;

  constexpr Eigen::Vector<double, 4> x_initial{{0.0, 0.0, 0.0, 0.0}};
  constexpr Eigen::Vector<double, 4> x_final{{1.0, std::numbers::pi, 0.0, 0.0}};

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
    X(0, k).SetValue(
        std::lerp(x_initial(0), x_final(0), static_cast<double>(k) / N));
    X(1, k).SetValue(
        std::lerp(x_initial(1), x_final(1), static_cast<double>(k) / N));
  }

  // u = f_x
  auto U = problem.U();

  // Initial conditions
  problem.ConstrainInitialState(x_initial);

  // Final conditions
  problem.ConstrainFinalState(x_final);

  // Cart position constraints
  problem.SubjectTo(X.Row(0) >= 0.0);
  problem.SubjectTo(X.Row(0) <= d_max.value());

  // Input constraints
  problem.SetLowerInputBound(-u_max.value());
  problem.SetUpperInputBound(u_max.value());

  // Minimize sum squared inputs
  sleipnir::Variable J = 0.0;
  for (int k = 0; k < N; ++k) {
    J += U.Col(k).T() * U.Col(k);
  }
  problem.Minimize(J);

  auto end1 = std::chrono::system_clock::now();
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  fmt::print("Setup time: {} ms\n\n",
             duration_cast<microseconds>(end1 - start).count() / 1000.0);

  auto status = problem.Solve({.diagnostics = true});

  CHECK(status.costFunctionType == sleipnir::ExpressionType::kQuadratic);
  CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kNonlinear);
  CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kLinear);

#if defined(__APPLE__) && defined(__aarch64__)
  // FIXME: Fails on macOS arm64 with "feasibility restoration failed"
  CHECK(status.exitCondition ==
        sleipnir::SolverExitCondition::kFeasibilityRestorationFailed);
  SKIP("Fails on macOS arm64 with \"feasibility restoration failed\"");
#else
  // FIXME: Fails on other platforms with "locally infeasible"
  CHECK(status.exitCondition ==
        sleipnir::SolverExitCondition::kLocallyInfeasible);
  SKIP("Fails with \"locally infeasible\"");
#endif

  // Verify initial state
  CHECK(X.Value(0, 0) == Catch::Approx(x_initial(0)).margin(1e-8));
  CHECK(X.Value(1, 0) == Catch::Approx(x_initial(1)).margin(1e-8));
  CHECK(X.Value(2, 0) == Catch::Approx(x_initial(2)).margin(1e-8));
  CHECK(X.Value(3, 0) == Catch::Approx(x_initial(3)).margin(1e-8));

  // Verify solution
  Eigen::Matrix<double, 4, 1> x{0.0, 0.0, 0.0, 0.0};
  Eigen::Matrix<double, 1, 1> u{0.0};
  for (int k = 0; k < N; ++k) {
    // Cart position constraints
    CHECK(X(0, k) >= 0.0);
    CHECK(X(0, k) <= d_max.value());

    // Input constraints
    CHECK(U(0, k) >= -u_max.value());
    CHECK(U(0, k) <= u_max.value());

    // Verify state
    CHECK(X.Value(0, k) == Catch::Approx(x(0)).margin(1e-2));
    CHECK(X.Value(1, k) == Catch::Approx(x(1)).margin(1e-2));
    CHECK(X.Value(2, k) == Catch::Approx(x(2)).margin(1e-2));
    CHECK(X.Value(3, k) == Catch::Approx(x(3)).margin(1e-2));
    INFO(fmt::format("  k = {}", k));

    // Project state forward
    x = RK4(CartPoleDynamicsDouble, x, u, dt);
  }

  // Verify final state
  CHECK(X.Value(0, N - 1) == Catch::Approx(x_final(0)).margin(1e-8));
  CHECK(X.Value(1, N - 1) == Catch::Approx(x_final(1)).margin(1e-8));
  CHECK(X.Value(2, N - 1) == Catch::Approx(x_final(2)).margin(1e-8));
  CHECK(X.Value(3, N - 1) == Catch::Approx(x_final(3)).margin(1e-8));

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
