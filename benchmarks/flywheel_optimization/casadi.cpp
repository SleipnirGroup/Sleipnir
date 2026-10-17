// Copyright (c) Sleipnir contributors

#include "casadi.hpp"

#include <chrono>
#include <cmath>
#include <exception>
#include <span>

#include <Eigen/Core>
#include <casadi/casadi.hpp>

#include "benchmark_util.hpp"

namespace {

/// Creates a flywheel quadratic optimization problem with CasADi.
///
/// @param dt Timestep duration.
/// @param N Number of samples in the problem.
casadi::Opti flywheel_casadi(std::chrono::duration<double> dt, int N) {
  // Flywheel model:
  // States: [velocity]
  // Inputs: [voltage]
  Eigen::Matrix<double, 1, 1> A{std::exp(-dt.count())};
  Eigen::Matrix<double, 1, 1> B{1.0 - std::exp(-dt.count())};

  casadi::MX ca_A = A(0, 0);
  casadi::MX ca_B = B(0, 0);

  casadi::Opti problem;
  casadi::Slice all;
  auto X = problem.variable(1, N + 1);
  auto U = problem.variable(1, N);

  // Dynamics constraint
  for (int k = 0; k < N; ++k) {
    problem.subject_to(X(all, k + 1) == ca_A * X(all, k) + ca_B * U(all, k));
  }

  // State and input constraints
  problem.subject_to(X(all, 0) == 0.0);
  problem.subject_to(problem.bounded(-12, U, 12));

  // Cost function - minimize error
  casadi::MX J = 0.0;
  for (int k = 0; k < N + 1; ++k) {
    J += ((10.0 - X(all, k)).T() * (10.0 - X(all, k)));
  }
  problem.minimize(J);

  return problem;
}

/// Solves a CasADi optimization problem.
///
/// @param problem The optimization problem.
/// @param diagnostics True if diagnostic prints should be enabled.
/// @return True if the solve succeeded.
bool solve(casadi::Opti& problem, bool diagnostics) {
  casadi::Dict plugin_options{{"expand", true}};
  casadi::Dict solver_options;

  // The problem is quadratic
  solver_options["hessian_constant"] = "yes";
  solver_options["jac_c_constant"] = "yes";
  solver_options["jac_d_constant"] = "yes";

  if (!diagnostics) {
    plugin_options["print_time"] = 0;
    solver_options["print_level"] = 0;
    solver_options["sb"] = "yes";
  }

  problem.solver("ipopt", plugin_options, solver_options);

  try {
    problem.solve();
  } catch (const std::exception&) {
    return false;
  }
  return true;
}

}  // namespace

void register_casadi_benchmarks(std::span<const int> nums_samples,
                                bool diagnostics) {
  register_optimization_benchmarks(
      "CasADi", nums_samples, &flywheel_casadi,
      [=](casadi::Opti& problem) { return solve(problem, diagnostics); });
}
