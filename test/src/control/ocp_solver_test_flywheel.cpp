// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <string>

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/control/ocp_solver.hpp>

#include "catch_string_converters.hpp"
#include "util/scope_exit.hpp"

using namespace std::chrono_literals;

namespace {
bool Near(double expected, double actual, double tolerance) {
  return std::abs(expected - actual) < tolerance;
}
}  // namespace

void TestFlywheel(std::string test_name, double A, double B,
                  const sleipnir::function_ref<sleipnir::VariableMatrix(
                      const sleipnir::VariableMatrix& x,
                      const sleipnir::VariableMatrix& u)>& F,
                  sleipnir::DynamicsType dynamics_type,
                  sleipnir::TranscriptionMethod method) {
  sleipnir::scope_exit exit{
      [] { CHECK(sleipnir::global_pool_resource().blocks_in_use() == 0u); }};

  constexpr std::chrono::duration<double> T = 5s;
  constexpr std::chrono::duration<double> dt = 5ms;
  constexpr int N = T / dt;

  // Flywheel model:
  // States: [velocity]
  // Inputs: [voltage]
  double A_discrete = std::exp(A * dt.count());
  double B_discrete = (1.0 - A_discrete) * B;

  constexpr double r = 10.0;

  sleipnir::OCPSolver solver(1, 1, dt, N, F, dynamics_type,
                             sleipnir::TimestepMethod::FIXED, method);
  solver.constrain_initial_state(0.0);
  solver.set_upper_input_bound(12);
  solver.set_lower_input_bound(-12);

  // Set up cost
  Eigen::Matrix<double, 1, N + 1> r_mat =
      Eigen::Matrix<double, 1, N + 1>::Constant(r);
  solver.minimize((r_mat - solver.X()) * (r_mat - solver.X()).T());

  auto status = solver.solve({.diagnostics = true});

  CHECK(status.cost_function_type == sleipnir::ExpressionType::QUADRATIC);
  CHECK(status.equality_constraint_type == sleipnir::ExpressionType::LINEAR);
  CHECK(status.inequality_constraint_type == sleipnir::ExpressionType::LINEAR);
  CHECK(status.exit_condition == sleipnir::SolverExitCondition::SUCCESS);

  // Voltage for steady-state velocity:
  //
  // rₖ₊₁ = Arₖ + Buₖ
  // uₖ = B⁺(rₖ₊₁ − Arₖ)
  // uₖ = B⁺(rₖ − Arₖ)
  // uₖ = B⁺(I − A)rₖ
  double u_ss = 1.0 / B_discrete * (1.0 - A_discrete) * r;

  // Verify initial state
  CHECK(solver.X().value(0, 0) == Catch::Approx(0.0).margin(1e-8));

  // Verify solution
  double x = 0.0;
  double u = 0.0;
  for (int k = 0; k < N; ++k) {
    // Verify state
    CHECK(solver.X().value(0, k) == Catch::Approx(x).margin(1e-2));

    // Determine expected input for this timestep
    double error = r - x;
    if (error > 1e-2) {
      // Max control input until the reference is reached
      u = 12.0;
    } else {
      // Maintain speed
      u = u_ss;
    }

    // Verify input
    if (k > 0 && k < N - 1 && Near(12.0, solver.U().value(0, k - 1), 1e-2) &&
        Near(u_ss, solver.U().value(0, k + 1), 1e-2)) {
      // If control input is transitioning between 12 and u_ss, ensure it's
      // within (u_ss, 12)
      CHECK(solver.U().value(0, k) >= u_ss);
      CHECK(solver.U().value(0, k) <= 12.0);
    } else {
      if (method == sleipnir::TranscriptionMethod::DIRECT_COLLOCATION) {
        // The tolerance is large because the trajectory is represented by a
        // spline, and splines chatter when transitioning quickly between
        // steady-states.
        CHECK(solver.U().value(0, k) == Catch::Approx(u).margin(2.0));
      } else {
        CHECK(solver.U().value(0, k) == Catch::Approx(u).margin(1e-4));
      }
    }

    INFO(std::format("  k = {}", k));

    // Project state forward
    x = A_discrete * x + B_discrete * u;
  }

  // Verify final state
  CHECK(solver.X().value(0, N) == Catch::Approx(r).margin(1e-7));

  // Log states for offline viewing
  std::ofstream states{std::format("{} states.csv", test_name)};
  if (states.is_open()) {
    states << "Time (s),Velocity (rad/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << std::format("{},{}\n", k * dt.count(), solver.X().value(0, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{std::format("{} inputs.csv", test_name)};
  if (inputs.is_open()) {
    inputs << "Time (s),Voltage (V)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << std::format("{},{}\n", k * dt.count(),
                              solver.U().value(0, k));
      } else {
        inputs << std::format("{},{}\n", k * dt.count(), 0.0);
      }
    }
  }
}

TEST_CASE("OCPSolver - Flywheel (explicit)", "[OCPSolver]") {
  constexpr double A = -1.0;
  constexpr double B = 1.0;

  auto f_ode = [=](sleipnir::VariableMatrix x, sleipnir::VariableMatrix u) {
    return A * x + B * u;
  };

  TestFlywheel("OCPSolver Flywheel Explicit Collocation", A, B, f_ode,
               sleipnir::DynamicsType::EXPLICIT_ODE,
               sleipnir::TranscriptionMethod::DIRECT_COLLOCATION);
  TestFlywheel("OCPSolver Flywheel Explicit Transcription", A, B, f_ode,
               sleipnir::DynamicsType::EXPLICIT_ODE,
               sleipnir::TranscriptionMethod::DIRECT_TRANSCRIPTION);
  TestFlywheel("OCPSolver Flywheel Explicit Single-Shooting", A, B, f_ode,
               sleipnir::DynamicsType::EXPLICIT_ODE,
               sleipnir::TranscriptionMethod::SINGLE_SHOOTING);
}

TEST_CASE("OCPSolver - Flywheel (discrete)", "[OCPSolver]") {
  constexpr double A = -1.0;
  constexpr double B = 1.0;
  constexpr std::chrono::duration<double> dt = 5ms;

  double A_discrete = std::exp(A * dt.count());
  double B_discrete = (1.0 - A_discrete) * B;

  auto f_discrete = [=](sleipnir::VariableMatrix x,
                        sleipnir::VariableMatrix u) {
    return A_discrete * x + B_discrete * u;
  };

  TestFlywheel("OCPSolver Flywheel Discrete Transcription", A, B, f_discrete,
               sleipnir::DynamicsType::DISCRETE,
               sleipnir::TranscriptionMethod::DIRECT_TRANSCRIPTION);
  TestFlywheel("OCPSolver Flywheel Discrete Single-Shooting", A, B, f_discrete,
               sleipnir::DynamicsType::DISCRETE,
               sleipnir::TranscriptionMethod::SINGLE_SHOOTING);
}
