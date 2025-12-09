// Copyright (c) Sleipnir contributors

#include "casadi.hpp"

#include <cmath>

#include <Eigen/Core>

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
