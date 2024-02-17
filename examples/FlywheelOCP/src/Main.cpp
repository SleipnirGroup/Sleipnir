// Copyright (c) Sleipnir contributors

#include <chrono>
#include <cmath>
#include <print>

#include <Eigen/Core>
#include <sleipnir/control/OCPSolver.hpp>
#include <sleipnir/optimization/OptimizationProblem.hpp>

#ifndef RUNNING_TESTS
int main() {
  using namespace std::chrono_literals;

  constexpr std::chrono::duration<double> T = 5s;
  constexpr std::chrono::duration<double> dt = 5ms;
  constexpr int N = T / dt;

  // Flywheel model:
  // States: [velocity]
  // Inputs: [voltage]
  Eigen::Matrix<double, 1, 1> A{-1.0};
  Eigen::Matrix<double, 1, 1> B{1.0};

  Eigen::Matrix<double, 1, 1> A_discrete{std::exp(A(0) * dt.count())};
  Eigen::Matrix<double, 1, 1> B_discrete{(1.0 - A_discrete(0)) * B(0)};

  auto f_discrete = [=](sleipnir::Variable t, sleipnir::VariableMatrix x,
                        sleipnir::VariableMatrix u, sleipnir::Variable dt) {
    return A_discrete * x + B_discrete * u;
  };

  Eigen::Matrix<double, 1, 1> r{10.0};

  sleipnir::OCPSolver solver(
      1, 1, dt, N, f_discrete, sleipnir::DynamicsType::kDiscrete,
      sleipnir::TimestepMethod::kFixed,
      sleipnir::TranscriptionMethod::kDirectTranscription);
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

  solver.Solve();

  // The first state
  std::println("x₀ = {}", solver.X().Value(0, 0));

  // The first input
  std::println("u₀ = {}", solver.U().Value(0, 0));
}
#endif
