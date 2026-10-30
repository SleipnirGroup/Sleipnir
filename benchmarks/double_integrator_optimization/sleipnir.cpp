// Copyright (c) Sleipnir contributors

#include "sleipnir.hpp"

#include <chrono>
#include <span>

#include <Eigen/Core>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>

#include "benchmark_util.hpp"

namespace {

/// Creates a double integrator convex nonlinear optimization problem with
/// Sleipnir.
///
/// The problem is a minimum-fuel rendezvous of a 2D point mass. The dynamics
/// are linear, the thrust magnitude constraint is convex, and the smoothed fuel
/// cost is strictly convex in the inputs, so the solution is unique.
///
/// @param dt Timestep duration.
/// @param N Number of samples in the problem.
slp::Problem<double> double_integrator_sleipnir(
    std::chrono::duration<double> dt, int N) {
  constexpr double a_max = 3.0;  // m/s²
  constexpr double ε = 0.1;      // Fuel cost smoothing (m/s²)

  constexpr Eigen::Vector<double, 4> x_initial{{0.0, 0.0, 0.0, 0.0}};
  constexpr Eigen::Vector<double, 4> x_final{{10.0, 5.0, 0.0, 0.0}};

  // Double integrator model:
  // States: [x, y, ẋ, ẏ]
  // Inputs: [ẍ, ÿ]
  double T = dt.count();
  Eigen::Matrix<double, 4, 4> A{{1.0, 0.0, T, 0.0},
                                {0.0, 1.0, 0.0, T},
                                {0.0, 0.0, 1.0, 0.0},
                                {0.0, 0.0, 0.0, 1.0}};
  Eigen::Matrix<double, 4, 2> B{
      {0.5 * T * T, 0.0}, {0.0, 0.5 * T * T}, {T, 0.0}, {0.0, T}};

  slp::Problem<double> problem;
  auto X = problem.decision_variable(4, N + 1);
  auto U = problem.decision_variable(2, N);

  // Dynamics constraint
  for (int k = 0; k < N; ++k) {
    problem.subject_to(X.col(k + 1) == A * X.col(k) + B * U.col(k));
  }

  // Initial and final conditions
  problem.subject_to(X.col(0) == x_initial);
  problem.subject_to(X.col(N) == x_final);

  // Thrust magnitude constraint
  for (int k = 0; k < N; ++k) {
    problem.subject_to(U[0, k] * U[0, k] + U[1, k] * U[1, k] <= a_max * a_max);
  }

  // Cost function - minimize smoothed fuel use
  slp::Variable J = 0.0;
  for (int k = 0; k < N; ++k) {
    J += slp::sqrt(U[0, k] * U[0, k] + U[1, k] * U[1, k] + ε * ε);
  }
  problem.minimize(J);

  return problem;
}

}  // namespace

void register_sleipnir_benchmarks(std::span<const int> nums_samples,
                                  bool diagnostics) {
  register_optimization_benchmarks(
      "Sleipnir", nums_samples, &double_integrator_sleipnir,
      [=](slp::Problem<double>& problem) {
        return problem.solve({.diagnostics = diagnostics}) ==
               slp::ExitStatus::SUCCESS;
      });
}
