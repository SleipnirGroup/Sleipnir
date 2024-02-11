// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <fstream>

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/core.h>
#include <sleipnir/control/OCPSolver.hpp>
#include <units/acceleration.h>
#include <units/angle.h>
#include <units/length.h>
#include <units/time.h>
#include <units/velocity.h>
#include <units/voltage.h>

#include "DifferentialDriveUtil.hpp"
#include "RK4.hpp"

TEST_CASE("OCPSolver - Differential drive", "[OCPSolver]") {
  auto start = std::chrono::system_clock::now();

  constexpr int N = 50;

  auto dynamics = [=](sleipnir::Variable t, sleipnir::VariableMatrix x,
                      sleipnir::VariableMatrix u, sleipnir::Variable dt) {
    // x = [x, y, heading, left velocity, right velocity]ᵀ
    // u = [left voltage, right voltage]ᵀ
    constexpr double trackwidth = (0.699_m).value();
    constexpr double Kv_linear = (3.02_V / 1_mps).value();
    constexpr double Ka_linear = (0.642_V / 1_mps_sq).value();
    constexpr double Kv_angular = (1.382_V / 1_mps).value();
    constexpr double Ka_angular = (0.08495_V / 1_mps_sq).value();

    auto v = (x(3) + x(4)) / 2.0;

    constexpr double A1 =
        -(Kv_linear / Ka_linear + Kv_angular / Ka_angular) / 2.0;
    constexpr double A2 =
        -(Kv_linear / Ka_linear - Kv_angular / Ka_angular) / 2.0;
    constexpr double B1 = 0.5 / Ka_linear + 0.5 / Ka_angular;
    constexpr double B2 = 0.5 / Ka_linear - 0.5 / Ka_angular;
    Eigen::Matrix<double, 2, 2> A{{A1, A2}, {A2, A1}};
    Eigen::Matrix<double, 2, 2> B{{B1, B2}, {B2, B1}};

    sleipnir::VariableMatrix xdot{5};
    xdot(0) = v * sleipnir::cos(x(2));
    xdot(1) = v * sleipnir::sin(x(2));
    xdot(2) = (x(4) - x(3)) / trackwidth;
    xdot.Segment(3, 2) = A * x.Segment(3, 2) + B * u;
    return xdot;
  };

  constexpr units::second_t minTimestep = 50_ms;
  constexpr Eigen::Vector<double, 5> x_initial{{0.0, 0.0, 0.0, 0.0, 0.0}};
  constexpr Eigen::Vector<double, 5> x_final{{1.0, 1.0, 0.0, 0.0, 0.0}};
  constexpr Eigen::Matrix<double, 2, 1> u_min{{-12.0, -12.0}};
  constexpr Eigen::Matrix<double, 2, 1> u_max{{12.0, 12.0}};

  sleipnir::OCPSolver problem(
      5, 2, std::chrono::duration<double>{minTimestep.value()}, N, dynamics,
      sleipnir::DynamicsType::kExplicitODE,
      sleipnir::TimestepMethod::kVariableSingle,
      sleipnir::TranscriptionMethod::kDirectTranscription);

  // Seed the min time formulation with lerp between waypoints
  for (int i = 0; i < N + 1; ++i) {
    problem.X()(0, i).SetValue(static_cast<double>(i) / (N + 1));
    problem.X()(1, i).SetValue(static_cast<double>(i) / (N + 1));
  }

  problem.ConstrainInitialState(x_initial);
  problem.ConstrainFinalState(x_final);

  problem.SetLowerInputBound(u_min);
  problem.SetUpperInputBound(u_max);

  // TODO: Solver is unhappy when more than one minimum timestep is constrained.
  // Detect this in either OptimizationProblem or OCPSolver.
  problem.SetMinTimestep(std::chrono::duration<double>{minTimestep.value()});
  problem.SetMaxTimestep(std::chrono::duration<double>{3.0});

  // Set up objective
  problem.Minimize(problem.DT() * Eigen::Matrix<double, N + 1, 1>::Ones());

  auto end1 = std::chrono::system_clock::now();
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  fmt::print("Setup time: {} ms\n\n",
             duration_cast<microseconds>(end1 - start).count() / 1000.0);

  auto status = problem.Solve({.maxIterations = 1000, .diagnostics = true});

  CHECK(status.costFunctionType == sleipnir::ExpressionType::kLinear);
  CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kNonlinear);
  CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kLinear);
  CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);

  auto X = problem.X();
  auto U = problem.U();

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
    CHECK(U(0, k).Value() >= -u_max(0));
    CHECK(U(0, k).Value() <= u_max(0));
    CHECK(U(1, k).Value() >= -u_max(1));
    CHECK(U(1, k).Value() <= u_max(1));

    // Verify state
    CHECK(X.Value(0, k) == Catch::Approx(x(0)).margin(1e-8));
    CHECK(X.Value(1, k) == Catch::Approx(x(1)).margin(1e-8));
    CHECK(X.Value(2, k) == Catch::Approx(x(2)).margin(1e-8));
    CHECK(X.Value(3, k) == Catch::Approx(x(3)).margin(1e-8));
    CHECK(X.Value(4, k) == Catch::Approx(x(4)).margin(1e-8));

    INFO(fmt::format("  k = {}", k));

    // Project state forward
    x = RK4(DifferentialDriveDynamicsDouble, x, u,
            units::second_t{problem.DT().Value(0, k)});
  }

  // Verify final state
  CHECK(X.Value(0, N) == Catch::Approx(x_final(0)).margin(1e-8));
  CHECK(X.Value(1, N) == Catch::Approx(x_final(1)).margin(1e-8));
  CHECK(X.Value(2, N) == Catch::Approx(x_final(2)).margin(1e-8));
  CHECK(X.Value(3, N) == Catch::Approx(x_final(3)).margin(1e-8));
  CHECK(X.Value(4, N) == Catch::Approx(x_final(4)).margin(1e-8));

  // Log states for offline viewing
  std::ofstream states{"OCPSolver Differential drive states.csv"};
  if (states.is_open()) {
    states << "Time (s),X position (m),Y position (m),Heading (rad),Left "
              "velocity (m/s),Right velocity (m/s)\n";

    double time = 0.0;
    for (int k = 0; k < N + 1; ++k) {
      states << fmt::format("{},{},{},{},{},{}\n", time,
                            problem.X().Value(0, k), problem.X().Value(1, k),
                            problem.X().Value(2, k), problem.X().Value(3, k),
                            problem.X().Value(4, k));

      time += problem.DT().Value(0, k);
    }
  }

  // Log inputs for offline viewing
  std::ofstream inputs{"OCPSolver Differential drive inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Left voltage (V),Right voltage (V)\n";

    double time = 0.0;
    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << fmt::format("{},{},{}\n", time, problem.U().Value(0, k),
                              problem.U().Value(1, k));
      } else {
        inputs << fmt::format("{},{},{}\n", time, 0.0, 0.0);
      }

      time += problem.DT().Value(0, k);
    }
  }
}
