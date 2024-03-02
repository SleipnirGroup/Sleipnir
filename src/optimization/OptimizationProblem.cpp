// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/OptimizationProblem.hpp"

#include <algorithm>
#include <array>
#include <iterator>

#include "optimization/solver/InteriorPoint.hpp"
#include "sleipnir/optimization/SolverExitCondition.hpp"
#include "util/Print.hpp"

namespace sleipnir {

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
  status.costFunctionType = m_f.value().Type();
}

void OptimizationProblem::Minimize(Variable&& cost) {
  m_f = std::move(cost);
  status.costFunctionType = m_f.value().Type();
}

void OptimizationProblem::Maximize(const Variable& cost) {
  // Maximizing a cost function is the same as minimizing its negative
  m_f = -cost;
  status.costFunctionType = m_f.value().Type();
}

void OptimizationProblem::Maximize(Variable&& cost) {
  // Maximizing a cost function is the same as minimizing its negative
  m_f = -std::move(cost);
  status.costFunctionType = m_f.value().Type();
}

void OptimizationProblem::SubjectTo(const EqualityConstraints& constraint) {
  // Get the highest order equality constraint expression type
  for (const auto& c : constraint.constraints) {
    status.equalityConstraintType =
        std::max(status.equalityConstraintType, c.Type());
  }

  m_equalityConstraints.reserve(m_equalityConstraints.size() +
                                constraint.constraints.size());
  std::copy(constraint.constraints.begin(), constraint.constraints.end(),
            std::back_inserter(m_equalityConstraints));
}

void OptimizationProblem::SubjectTo(EqualityConstraints&& constraint) {
  // Get the highest order equality constraint expression type
  for (const auto& c : constraint.constraints) {
    status.equalityConstraintType =
        std::max(status.equalityConstraintType, c.Type());
  }

  m_equalityConstraints.reserve(m_equalityConstraints.size() +
                                constraint.constraints.size());
  std::copy(constraint.constraints.begin(), constraint.constraints.end(),
            std::back_inserter(m_equalityConstraints));
}

void OptimizationProblem::SubjectTo(const InequalityConstraints& constraint) {
  // Get the highest order inequality constraint expression type
  for (const auto& c : constraint.constraints) {
    status.inequalityConstraintType =
        std::max(status.inequalityConstraintType, c.Type());
  }

  m_inequalityConstraints.reserve(m_inequalityConstraints.size() +
                                  constraint.constraints.size());
  std::copy(constraint.constraints.begin(), constraint.constraints.end(),
            std::back_inserter(m_inequalityConstraints));
}

void OptimizationProblem::SubjectTo(InequalityConstraints&& constraint) {
  // Get the highest order inequality constraint expression type
  for (const auto& c : constraint.constraints) {
    status.inequalityConstraintType =
        std::max(status.inequalityConstraintType, c.Type());
  }

  m_inequalityConstraints.reserve(m_inequalityConstraints.size() +
                                  constraint.constraints.size());
  std::copy(constraint.constraints.begin(), constraint.constraints.end(),
            std::back_inserter(m_inequalityConstraints));
}

SolverStatus OptimizationProblem::Solve(const SolverConfig& config) {
  // Create the initial value column vector
  Eigen::VectorXd x{m_decisionVariables.size()};
  for (size_t i = 0; i < m_decisionVariables.size(); ++i) {
    x(i) = m_decisionVariables[i].Value();
  }

  status.exitCondition = SolverExitCondition::kSuccess;

  // If there's no cost function, make it zero and continue
  if (!m_f.has_value()) {
    m_f = Variable();
  }

  if (config.diagnostics) {
    constexpr std::array kExprTypeToName{"empty", "constant", "linear",
                                         "quadratic", "nonlinear"};

    // Print cost function and constraint expression types
    sleipnir::println(
        "The cost function is {}.",
        kExprTypeToName[static_cast<int>(status.costFunctionType)]);
    sleipnir::println(
        "The equality constraints are {}.",
        kExprTypeToName[static_cast<int>(status.equalityConstraintType)]);
    sleipnir::println(
        "The inequality constraints are {}.",
        kExprTypeToName[static_cast<int>(status.inequalityConstraintType)]);
    sleipnir::println("");

    // Print problem dimensionality
    sleipnir::println("Number of decision variables: {}",
                      m_decisionVariables.size());
    sleipnir::println("Number of equality constraints: {}",
                      m_equalityConstraints.size());
    sleipnir::println("Number of inequality constraints: {}\n",
                      m_inequalityConstraints.size());
  }

  // If the problem is empty or constant, there's nothing to do
  if (status.costFunctionType <= ExpressionType::kConstant &&
      status.equalityConstraintType <= ExpressionType::kConstant &&
      status.inequalityConstraintType <= ExpressionType::kConstant) {
    return status;
  }

  // Solve the optimization problem
  Eigen::VectorXd s = Eigen::VectorXd::Ones(m_inequalityConstraints.size());
  InteriorPoint(m_decisionVariables, m_equalityConstraints,
                m_inequalityConstraints, m_f.value(), m_callback, config, false,
                x, s, &status);

  if (config.diagnostics) {
    sleipnir::println("Exit condition: {}", ToMessage(status.exitCondition));
  }

  // Assign the solution to the original Variable instances
  VariableMatrix{m_decisionVariables}.SetValue(x);

  return status;
}

}  // namespace sleipnir
