// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <fstream>

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/core.h>
#include <sleipnir/control/OCPSolver.hpp>
#include <sleipnir/optimization/OptimizationProblem.hpp>

using namespace std::chrono_literals;

namespace {
bool Near(double expected, double actual, double tolerance) {
  return std::abs(expected - actual) < tolerance;
}
}  // namespace

void TestFlywheel(std::string testName, double A, double B,
                  const sleipnir::DynamicsFunction& F,
                  sleipnir::DynamicsType dynamicsType,
                  sleipnir::TranscriptionMethod method) {
  constexpr std::chrono::duration<double> T = 5s;
  constexpr std::chrono::duration<double> dt = 5ms;
  constexpr int N = T / dt;

  // Flywheel model:
  // States: [velocity]
  // Inputs: [voltage]
  double A_discrete = std::exp(A * dt.count());
  double B_discrete = (1.0 - A_discrete) * B;
  constexpr double r = 10.0;

  sleipnir::OCPSolver solver(1, 1, dt, N, F, dynamicsType,
                             sleipnir::TimestepMethod::kFixed, method);
  solver.ConstrainInitialState(0.0);
  solver.SetUpperInputBound(12);
  solver.SetLowerInputBound(-12);

  // Set up objective
  Eigen::Matrix<double, 1, N + 1> r_mat =
      r * Eigen::Matrix<double, 1, N + 1>::Ones();
  sleipnir::VariableMatrix r_mat_vmat{r_mat};
  sleipnir::VariableMatrix objective =
      (r_mat_vmat - solver.X()) * (r_mat_vmat - solver.X()).T();
  solver.Minimize(objective);

  auto status = solver.Solve({.diagnostics = true});

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
  double u_ss = 1.0 / B_discrete * (1.0 - A_discrete) * r;

  // Verify initial state
  CHECK(solver.X().Value(0, 0) == Catch::Approx(0.0).margin(1e-8));

  // Verify solution
  double x = 0.0;
  double u = 0.0;
  for (int k = 0; k < N; ++k) {
    // Verify state
    CHECK(solver.X().Value(0, k) == Catch::Approx(x).margin(1e-2));

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
    if (k > 0 && k < N - 1 && Near(12.0, solver.U().Value(0, k - 1), 1e-2) &&
        Near(u_ss, solver.U().Value(0, k + 1), 1e-2)) {
      // If control input is transitioning between 12 and u_ss, ensure it's
      // within (u_ss, 12)
      CHECK(solver.U().Value(0, k) >= u_ss);
      CHECK(solver.U().Value(0, k) <= 12.0);
    } else {
      CHECK(solver.U().Value(0, k) == Catch::Approx(u).margin(1.0));
    }

    INFO(fmt::format("  k = {}", k));

    // Project state forward
    x = A_discrete * x + B_discrete * u;
  }

  // Verify final state
  CHECK(solver.X().Value(0, N) == Catch::Approx(r).margin(1e-7));

  // Log states for offline viewing
  std::ofstream states{fmt::format("{} states.csv", testName)};
  if (states.is_open()) {
    states << "Time (s),Velocity (rad/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << fmt::format("{},{}\n", k * dt.count(), solver.X().Value(0, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{fmt::format("{} inputs.csv", testName)};
  if (inputs.is_open()) {
    inputs << "Time (s),Voltage (V)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << fmt::format("{},{}\n", k * dt.count(),
                              solver.U().Value(0, k));
      } else {
        inputs << fmt::format("{},{}\n", k * dt.count(), 0.0);
      }
    }
  }
}

TEST_CASE("OCPSolver - Flywheel (explicit)", "[OCPSolver]") {
  constexpr double A = -1.0;
  constexpr double B = 1.0;

  auto f_ode = [=](sleipnir::Variable t, sleipnir::VariableMatrix x,
                   sleipnir::VariableMatrix u,
                   sleipnir::Variable dt) { return A * x + B * u; };

  TestFlywheel("OCPSolver Flywheel Explicit Collocation", A, B, f_ode,
               sleipnir::DynamicsType::kExplicitODE,
               sleipnir::TranscriptionMethod::kDirectCollocation);
  TestFlywheel("OCPSolver Flywheel Explicit Transcription", A, B, f_ode,
               sleipnir::DynamicsType::kExplicitODE,
               sleipnir::TranscriptionMethod::kDirectTranscription);
  TestFlywheel("OCPSolver Flywheel Explicit Single-Shooting", A, B, f_ode,
               sleipnir::DynamicsType::kExplicitODE,
               sleipnir::TranscriptionMethod::kSingleShooting);
}

TEST_CASE("OCPSolver - Flywheel (discrete)", "[OCPSolver]") {
  constexpr double A = -1.0;
  constexpr double B = 1.0;
  constexpr std::chrono::duration<double> dt = 5ms;

  double A_discrete = std::exp(A * dt.count());
  double B_discrete = (1.0 - A_discrete) * B;

  auto f_discrete = [=](sleipnir::Variable t, sleipnir::VariableMatrix x,
                        sleipnir::VariableMatrix u, sleipnir::Variable dt) {
    return A_discrete * x + B_discrete * u;
  };

  TestFlywheel("OCPSolver Flywheel Discrete Transcription", A, B, f_discrete,
               sleipnir::DynamicsType::kDiscrete,
               sleipnir::TranscriptionMethod::kDirectTranscription);
  TestFlywheel("OCPSolver Flywheel Discrete Single-Shooting", A, B, f_discrete,
               sleipnir::DynamicsType::kDiscrete,
               sleipnir::TranscriptionMethod::kSingleShooting);
}
