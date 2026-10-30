// Copyright (c) Sleipnir contributors

#include "casadi.hpp"

#include <chrono>
#include <exception>
#include <span>
#include <vector>

#include <casadi/casadi.hpp>

#include "benchmark_util.hpp"

namespace {

/// Creates a double integrator convex nonlinear optimization problem with
/// CasADi.
///
/// The problem is a minimum-fuel rendezvous of a 2D point mass. The dynamics
/// are linear, the thrust magnitude constraint is convex, and the smoothed fuel
/// cost is strictly convex in the inputs, so the solution is unique.
///
/// @param dt Timestep duration.
/// @param N Number of samples in the problem.
casadi::Opti double_integrator_casadi(std::chrono::duration<double> dt, int N) {
  constexpr double a_max = 3.0;  // m/s²
  constexpr double ε = 0.1;      // Fuel cost smoothing (m/s²)

  // Double integrator model:
  // States: [x, y, ẋ, ẏ]
  // Inputs: [ẍ, ÿ]
  double T = dt.count();
  casadi::DM A{std::vector<std::vector<double>>{{1.0, 0.0, T, 0.0},
                                                {0.0, 1.0, 0.0, T},
                                                {0.0, 0.0, 1.0, 0.0},
                                                {0.0, 0.0, 0.0, 1.0}}};
  casadi::DM B{std::vector<std::vector<double>>{
      {0.5 * T * T, 0.0}, {0.0, 0.5 * T * T}, {T, 0.0}, {0.0, T}}};

  casadi::Opti problem;
  casadi::Slice all;
  auto X = problem.variable(4, N + 1);
  auto U = problem.variable(2, N);

  // Dynamics constraint
  for (int k = 0; k < N; ++k) {
    problem.subject_to(X(all, k + 1) == casadi::MX::mtimes(A, X(all, k)) +
                                            casadi::MX::mtimes(B, U(all, k)));
  }

  // Initial and final conditions
  problem.subject_to(X(all, 0) == casadi::DM{0.0, 0.0, 0.0, 0.0});
  problem.subject_to(X(all, N) == casadi::DM{10.0, 5.0, 0.0, 0.0});

  // Thrust magnitude constraint
  for (int k = 0; k < N; ++k) {
    problem.subject_to(U(0, k) * U(0, k) + U(1, k) * U(1, k) <= a_max * a_max);
  }

  // Cost function - minimize smoothed fuel use
  casadi::MX J = 0.0;
  for (int k = 0; k < N; ++k) {
    J += sqrt(U(0, k) * U(0, k) + U(1, k) * U(1, k) + ε * ε);
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
      "CasADi", nums_samples, &double_integrator_casadi,
      [=](casadi::Opti& problem) { return solve(problem, diagnostics); });
}
