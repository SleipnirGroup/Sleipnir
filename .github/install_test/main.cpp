// Copyright (c) Sleipnir contributors

#include <sleipnir/optimization/problem.hpp>

int main() {
  // Find the x, y pair with the largest product for which x + 3y = 36
  slp::Problem<double> problem;

  auto x = problem.decision_variable();
  auto y = problem.decision_variable();

  problem.maximize(x * y);
  problem.subject_to(x + 3 * y == 36);

  return problem.solve() == slp::ExitStatus::SUCCESS ? 0 : 1;
}
