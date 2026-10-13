// Copyright (c) Sleipnir contributors

#include "sleipnir.hpp"

#include <chrono>
#include <cmath>
#include <span>

#include <Eigen/Core>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>

#include "benchmark_util.hpp"

namespace {

/// Creates a flywheel quadratic optimization problem with Sleipnir.
///
/// @param dt Timestep duration.
/// @param N Number of samples in the problem.
slp::Problem<double> flywheel_sleipnir(std::chrono::duration<double> dt,
                                       int N) {
  // Flywheel model:
  // States: [velocity]
  // Inputs: [voltage]
  Eigen::Matrix<double, 1, 1> A{std::exp(-dt.count())};
  Eigen::Matrix<double, 1, 1> B{1.0 - std::exp(-dt.count())};

  slp::Problem<double> problem;
  auto X = problem.decision_variable(1, N + 1);
  auto U = problem.decision_variable(1, N);

  // Dynamics constraint
  for (int k = 0; k < N; ++k) {
    problem.subject_to(X.col(k + 1) == A * X.col(k) + B * U.col(k));
  }

  // State and input constraints
  problem.subject_to(X.col(0) == 0.0);
  problem.subject_to(slp::bounds(-12, U, 12));

  // Cost function - minimize error
  Eigen::Matrix<double, 1, 1> r{10.0};
  slp::Variable J = 0.0;
  for (int k = 0; k < N + 1; ++k) {
    J += ((r - X.col(k)).T() * (r - X.col(k)));
  }
  problem.minimize(J);

  return problem;
}

}  // namespace

void register_sleipnir_benchmarks(std::span<const int> nums_samples,
                                  bool diagnostics) {
  register_optimization_benchmarks(
      "Sleipnir", nums_samples, &flywheel_sleipnir,
      [=](slp::Problem<double>& problem) {
        return problem.solve({.diagnostics = diagnostics}) ==
               slp::ExitStatus::SUCCESS;
      });
}
