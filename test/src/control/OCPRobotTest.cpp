// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <fstream>

#include <Eigen/Core>
#include <fmt/core.h>
#include <gtest/gtest.h>
#include <sleipnir/control/OCPSolver.hpp>
#include <units/time.h>

TEST(OCPSolverTest, Robot) {
  auto start = std::chrono::system_clock::now();

  constexpr int N = 50;

  auto dynamicsFunction = [=](sleipnir::Variable t, sleipnir::VariableMatrix x,
                              sleipnir::VariableMatrix u,
                              sleipnir::Variable dt) {
    sleipnir::Variable theta = x(2, 0);
    sleipnir::Variable Vc = 0.5 * (u(0, 0) + u(1, 0));
    sleipnir::Variable w = 0.5 * (u(0, 0) - u(1, 0));
    auto Vx = Vc * sleipnir::cos(theta);
    auto Vy = Vc * sleipnir::sin(theta);
    sleipnir::VariableMatrix xdot{3, 1};
    xdot(0, 0) = Vx;
    xdot(1, 0) = Vy;
    xdot(2, 0) = w;
    return xdot;
  };

  constexpr units::second_t minTimestep = 50_ms;
  constexpr Eigen::Matrix<double, 3, 1> initialState{{0.0, 0.0, 0.0}};
  constexpr Eigen::Matrix<double, 3, 1> finalState{{10.0, 10.0, 0.0}};
  constexpr Eigen::Matrix<double, 2, 1> inputMax{{1.0, 1.0}};
  constexpr Eigen::Matrix<double, 2, 1> inputMin{{0.0, 0.0}};

  sleipnir::OCPSolver solverMinTime(
      3, 2, std::chrono::duration<double>(minTimestep.value()), N,
      dynamicsFunction, sleipnir::DynamicsType::kExplicitODE,
      sleipnir::TimestepMethod::kVariableSingle,
      sleipnir::TranscriptionMethod::kDirectTranscription);

  // Seed the min time formulation with lerp between waypoints
  for (int i = 0; i < N + 1; ++i) {
    solverMinTime.X()(0, i).SetValue(static_cast<double>(i) / (N + 1));
    solverMinTime.X()(1, i).SetValue(static_cast<double>(i) / (N + 1));
  }

  solverMinTime.ConstrainInitialState(initialState);
  solverMinTime.ConstrainFinalState(finalState);

  solverMinTime.SetLowerInputBound(inputMin);
  solverMinTime.SetUpperInputBound(inputMax);

  // TODO: Solver is unhappy when more than one minimum timestep is constrained.
  // Detect this in either OptimizationProblem or OCPSolver.
  solverMinTime.SetMinTimestep(
      std::chrono::duration<double>(minTimestep.value()));
  solverMinTime.SetMaxTimestep(std::chrono::duration<double>(3.0));

  // Set up objective
  solverMinTime.Minimize(solverMinTime.DT() *
                         Eigen::Matrix<double, N + 1, 1>::Ones());

  auto end1 = std::chrono::system_clock::now();
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  fmt::print("Setup time: {} ms\n\n",
             duration_cast<microseconds>(end1 - start).count() / 1000.0);

  auto status =
      solverMinTime.Solve({.maxIterations = 1000, .diagnostics = true});

  EXPECT_EQ(sleipnir::ExpressionType::kLinear, status.costFunctionType);
  EXPECT_EQ(sleipnir::ExpressionType::kNonlinear,
            status.equalityConstraintType);
  EXPECT_EQ(sleipnir::ExpressionType::kLinear, status.inequalityConstraintType);
  // FIXME: Poor convergence
  // EXPECT_EQ(sleipnir::SolverExitCondition::kSuccess, status.exitCondition);

  // Log states for offline viewing
  std::ofstream states{"Robot states.csv"};
  double time = 0.0;
  if (states.is_open()) {
    states << "Time (s),X estimate (m),Y estimate (m),Theta (rad),X reference "
              "(m),Y reference (m)\n";

    for (int k = 0; k < N + 1; ++k) {
      states << fmt::format("{},{},{},{},{},{}\n", time,
                            solverMinTime.X().Value(0, k),
                            solverMinTime.X().Value(1, k),
                            solverMinTime.X().Value(2, k), 0.0, 0.0);
      time += solverMinTime.DT().Value(0, k);
    }
  }

  time = 0.0;
  // Log inputs for offline viewing
  std::ofstream inputs{"Robot inputs.csv"};
  if (inputs.is_open()) {
    inputs << "Time (s),Velocity Right (m/s),Velocity Left (m/s)\n";

    for (int k = 0; k < N + 1; ++k) {
      if (k < N) {
        inputs << fmt::format("{},{},{}\n", time, solverMinTime.U().Value(0, k),
                              solverMinTime.U().Value(1, k));
      } else {
        inputs << fmt::format("{},{},{}\n", time, 0.0, 0.0);
      }
      time += solverMinTime.DT().Value(0, k);
    }
  }
}
