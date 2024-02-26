// Copyright (c) Sleipnir contributors

#include "Sleipnir.hpp"

#include <cmath>

#include <Eigen/Core>

sleipnir::OptimizationProblem FlywheelSleipnir(std::chrono::duration<double> dt,
                                               int N) {
  // Flywheel model:
  // States: [velocity]
  // Inputs: [voltage]
  Eigen::Matrix<double, 1, 1> A{std::exp(-dt.count())};
  Eigen::Matrix<double, 1, 1> B{1.0 - std::exp(-dt.count())};

  sleipnir::OptimizationProblem problem;
  auto X = problem.DecisionVariable(1, N + 1);
  auto U = problem.DecisionVariable(1, N);

  // Dynamics constraint
  for (int k = 0; k < N; ++k) {
    problem.SubjectTo(X.Col(k + 1) == A * X.Col(k) + B * U.Col(k));
  }

  // State and input constraints
  problem.SubjectTo(X.Col(0) == 0.0);
  problem.SubjectTo(-12 <= U);
  problem.SubjectTo(U <= 12);

  // Cost function - minimize error
  Eigen::Matrix<double, 1, 1> r{10.0};
  sleipnir::Variable J = 0.0;
  for (int k = 0; k < N + 1; ++k) {
    J += ((r - X.Col(k)).T() * (r - X.Col(k)));
  }
  problem.Minimize(J);

  return problem;
}
