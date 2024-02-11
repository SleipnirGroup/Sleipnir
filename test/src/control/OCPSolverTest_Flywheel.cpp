// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/QR>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/core.h>
#include <sleipnir/control/OCPSolver.hpp>
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

void TestFlywheel(std::string testName, Eigen::Matrix<double, 1, 1> A,
                  Eigen::Matrix<double, 1, 1> B,
                  const sleipnir::DynamicsFunction& F,
                  sleipnir::DynamicsType dynamicsType,
                  sleipnir::TranscriptionMethod method) {
  auto start = std::chrono::system_clock::now();

  constexpr auto T = 5_s;
  constexpr units::second_t dt = 5_ms;
  constexpr int N = T / dt;

  // Flywheel model:
  // States: [velocity]
  // Inputs: [voltage]
  Eigen::Matrix<double, 1, 1> A_discrete{std::exp(A(0) * dt.value())};
  Eigen::Matrix<double, 1, 1> B_discrete{(1.0 - A_discrete(0)) * B(0)};
  Eigen::Matrix<double, 1, 1> r{10.0};

  sleipnir::OCPSolver solver(1, 1, std::chrono::duration<double>{dt.value()}, N,
                             F, dynamicsType, sleipnir::TimestepMethod::kFixed,
                             method);
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

  [[maybe_unused]] auto end1 = std::chrono::system_clock::now();
  if (Argv().Contains("--enable-diagnostics")) {
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    fmt::print("Setup time: {} ms\n\n",
               duration_cast<microseconds>(end1 - start).count() / 1000.0);
  }

  auto status =
      solver.Solve({.diagnostics = Argv().Contains("--enable-diagnostics")});

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
      B_discrete.householderQr().solve(decltype(A_discrete)::Identity() -
                                       A_discrete) *
      r;

  // Verify initial state
  CHECK(solver.X().Value(0, 0) == Catch::Approx(0.0).margin(1e-8));

  // Verify solution
  Eigen::Matrix<double, 1, 1> x{0.0};
  Eigen::Matrix<double, 1, 1> u{0.0};
  for (int k = 0; k < N; ++k) {
    // Verify state
    CHECK(solver.X().Value(0, k) == Catch::Approx(x(0)).margin(1e-2));

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
    if (k > 0 && k < N - 1 && Near(12.0, solver.U().Value(0, k - 1), 1e-2) &&
        Near(u_ss(0), solver.U().Value(0, k + 1), 1e-2)) {
      // If control input is transitioning between 12 and u_ss, ensure it's
      // within (u_ss, 12)
      CHECK(solver.U().Value(0, k) >= u_ss(0));
      CHECK(solver.U().Value(0, k) <= 12.0);
    } else {
      CHECK(solver.U().Value(0, k) == Catch::Approx(u(0)).margin(1.0));
    }

    INFO(fmt::format("  k = {}", k));

    // Project state forward
    x = A_discrete * x + B_discrete * u;
  }

  // Verify final state
  CHECK(solver.X().Value(0, N) == Catch::Approx(r(0)).margin(1e-7));

  // Log states for offline viewing
  std::ofstream states{fmt::format("{} states.csv", testName)};
  if (states.is_open()) {
    states << "Time (s),Velocity (rad/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << fmt::format("{},{}\n", k * units::second_t{dt}.value(),
                            solver.X().Value(0, k));
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{fmt::format("{} inputs.csv", testName)};
  if (inputs.is_open()) {
    inputs << "Time (s),Voltage (V)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << fmt::format("{},{}\n", k * dt.value(),
                              solver.U().Value(0, k));
      } else {
        inputs << fmt::format("{},{}\n", k * dt.value(), 0.0);
      }
    }
  }
}

TEST_CASE("Flywheel (explicit)", "[OCPSolver]") {
  Eigen::Matrix<double, 1, 1> A{-1.0};
  Eigen::Matrix<double, 1, 1> B{1.0};

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

TEST_CASE("Flywheel (discrete)", "[OCPSolver]") {
  Eigen::Matrix<double, 1, 1> A{-1.0};
  Eigen::Matrix<double, 1, 1> B{1.0};
  constexpr units::second_t dt = 5_ms;

  Eigen::Matrix<double, 1, 1> A_discrete{std::exp(A(0) * dt.value())};
  Eigen::Matrix<double, 1, 1> B_discrete{(1.0 - A_discrete(0)) * B(0)};

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
