// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/OptimizationProblem.hpp"

#include <array>

#include <fmt/core.h>

#include "optimization/solver/InteriorPoint.hpp"

using namespace sleipnir;

OptimizationProblem::OptimizationProblem() noexcept {
  m_decisionVariables.reserve(1024);
  m_equalityConstraints.reserve(1024);
  m_inequalityConstraints.reserve(1024);
}

Variable OptimizationProblem::DecisionVariable() {
  m_decisionVariables.emplace_back();
  return m_decisionVariables.back();
}

VariableMatrix OptimizationProblem::DecisionVariable(int rows, int cols) {
  m_decisionVariables.reserve(m_decisionVariables.size() + rows * cols);

  VariableMatrix vars{rows, cols};

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      m_decisionVariables.emplace_back();
      vars(row, col) = m_decisionVariables.back();
    }
  }

  return vars;
}

VariableMatrix OptimizationProblem::SymmetricDecisionVariable(int rows) {
  // We only need to store the lower triangle of an n x n symmetric matrix; the
  // other elements are duplicates. The lower triangle has (n² + n)/2 elements.
  //
  //   n
  //   Σ k = (n² + n)/2
  //  k=1
  m_decisionVariables.reserve(m_decisionVariables.size() +
                              (rows * rows + rows) / 2);

  VariableMatrix vars{rows, rows};

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col <= row; ++col) {
      m_decisionVariables.emplace_back();
      vars(row, col) = m_decisionVariables.back();
      vars(col, row) = m_decisionVariables.back();
    }
  }

  return vars;
}

void OptimizationProblem::Minimize(const Variable& cost) {
  m_f = cost;
}

void OptimizationProblem::Minimize(Variable&& cost) {
  m_f = std::move(cost);
}

void OptimizationProblem::Maximize(const Variable& cost) {
  // Maximizing a cost function is the same as minimizing its negative
  m_f = -cost;
}

void OptimizationProblem::Maximize(Variable&& cost) {
  // Maximizing a cost function is the same as minimizing its negative
  m_f = -std::move(cost);
}

void OptimizationProblem::SubjectTo(const EqualityConstraints& constraint) {
  auto& storage = constraint.constraints;

  m_equalityConstraints.reserve(m_equalityConstraints.size() + storage.size());

  for (size_t i = 0; i < storage.size(); ++i) {
    m_equalityConstraints.emplace_back(storage[i]);
  }
}

void OptimizationProblem::SubjectTo(EqualityConstraints&& constraint) {
  auto& storage = constraint.constraints;

  m_equalityConstraints.reserve(m_equalityConstraints.size() + storage.size());

  for (size_t i = 0; i < storage.size(); ++i) {
    m_equalityConstraints.emplace_back(std::move(storage[i]));
  }
}

void OptimizationProblem::SubjectTo(const InequalityConstraints& constraint) {
  auto& storage = constraint.constraints;

  m_inequalityConstraints.reserve(m_inequalityConstraints.size() +
                                  storage.size());

  for (size_t i = 0; i < storage.size(); ++i) {
    m_inequalityConstraints.emplace_back(storage[i]);
  }
}

void OptimizationProblem::SubjectTo(InequalityConstraints&& constraint) {
  auto& storage = constraint.constraints;

  m_inequalityConstraints.reserve(m_inequalityConstraints.size() +
                                  storage.size());

  for (size_t i = 0; i < storage.size(); ++i) {
    m_inequalityConstraints.emplace_back(std::move(storage[i]));
  }
}

SolverStatus OptimizationProblem::Solve(const SolverConfig& config) {
  // Create the initial value column vector
  Eigen::VectorXd x{m_decisionVariables.size()};
  for (size_t i = 0; i < m_decisionVariables.size(); ++i) {
    x(i) = m_decisionVariables[i].Value();
  }

  SolverStatus status;

  if (m_f.has_value()) {
    // If there's a cost function, get its expression type. The default is
    // "none".
    status.costFunctionType = m_f.value().Type();
  } else {
    // If there's no cost function, make it zero and continue
    m_f = Variable();
  }

  // Get the highest order equality constraint expression type
  for (const auto& constraint : m_equalityConstraints) {
    auto constraintType = constraint.Type();
    if (status.equalityConstraintType < constraintType) {
      status.equalityConstraintType = constraintType;
    }
  }

  // Get the highest order inequality constraint expression type
  for (const auto& constraint : m_inequalityConstraints) {
    auto constraintType = constraint.Type();
    if (status.inequalityConstraintType < constraintType) {
      status.inequalityConstraintType = constraintType;
    }
  }

  if (config.diagnostics) {
    constexpr std::array<const char*, 5> kExprTypeToName = {
        "empty", "constant", "linear", "quadratic", "nonlinear"};

    // Print cost function and constraint expression types
    fmt::print("The cost function is {}.\n",
               kExprTypeToName[static_cast<int>(status.costFunctionType)]);
    fmt::print(
        "The equality constraints are {}.\n",
        kExprTypeToName[static_cast<int>(status.equalityConstraintType)]);
    fmt::print(
        "The inequality constraints are {}.\n",
        kExprTypeToName[static_cast<int>(status.inequalityConstraintType)]);
    fmt::print("\n");

    // Print problem dimensionality
    fmt::print("Number of decision variables: {}\n",
               m_decisionVariables.size());
    fmt::print("Number of equality constraints: {}\n",
               m_equalityConstraints.size());
    fmt::print("Number of inequality constraints: {}\n\n",
               m_inequalityConstraints.size());
  }

  // If the problem is empty or constant, there's nothing to do
  if (status.costFunctionType <= ExpressionType::kConstant &&
      status.equalityConstraintType <= ExpressionType::kConstant &&
      status.inequalityConstraintType <= ExpressionType::kConstant) {
    return status;
  }

  // Solve the optimization problem
  Eigen::VectorXd solution =
      InteriorPoint(m_decisionVariables, m_f, m_equalityConstraints,
                    m_inequalityConstraints, m_callback, config, x, &status);

  if (config.diagnostics) {
    PrintExitCondition(status.exitCondition);
  }

  // Assign the solution to the original Variable instances
  VariableMatrix{m_decisionVariables}.SetValue(solution);

  return status;
}

void OptimizationProblem::PrintExitCondition(
    const SolverExitCondition& exitCondition) {
  using enum SolverExitCondition;

  fmt::print("Exit condition: ");
  switch (exitCondition) {
    case kSuccess:
      fmt::print("solved to desired tolerance");
      break;
    case kSolvedToAcceptableTolerance:
      fmt::print("solved to acceptable tolerance");
      break;
    case kCallbackRequestedStop:
      fmt::print("callback requested stop");
      break;
    case kTooFewDOFs:
      fmt::print("problem has too few degrees of freedom");
      break;
    case kLocallyInfeasible:
      fmt::print("problem is locally infeasible");
      break;
    case kBadSearchDirection:
      fmt::print(
          "solver failed to reach the desired tolerance due to a bad search "
          "direction");
      break;
    case kMaxSearchDirectionTooSmall:
      fmt::print(
          "solver failed to reach the desired tolerance due to the maximum "
          "search direction becoming too small");
      break;
    case kDivergingIterates:
      fmt::print(
          "solver encountered diverging primal iterates pₖˣ and/or pₖˢ and "
          "gave up");
      break;
    case kMaxIterationsExceeded:
      fmt::print("solution returned after maximum iterations exceeded");
      break;
    case kMaxWallClockTimeExceeded:
      fmt::print("solution returned after maximum wall time exceeded");
      break;
  }
  fmt::print("\n");
}
