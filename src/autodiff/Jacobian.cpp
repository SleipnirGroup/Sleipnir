// Copyright (c) Joshua Nichols and Tyler Veness

#include "sleipnir/autodiff/Jacobian.hpp"

#include "sleipnir/autodiff/Gradient.hpp"

using namespace sleipnir::autodiff;

Jacobian::Jacobian(VectorXvar variables, VectorXvar wrt) noexcept
    : m_variables{std::move(variables)}, m_wrt{std::move(wrt)} {
  m_profiler.StartSetup();

  std::vector<Expression*> row;
  std::vector<Expression*> stack;
  for (Variable variable : m_variables) {
    // BFS
    row.clear();

    stack.emplace_back(variable.expr.Get());

    // Initialize the number of instances of each node in the tree
    // (Expression::duplications)
    while (!stack.empty()) {
      auto& currentNode = stack.back();
      stack.pop_back();

      for (auto&& arg : currentNode->args) {
        // Only continue if the node is not a constant and hasn't already been
        // explored.
        if (arg != nullptr && arg->type != ExpressionType::kConstant) {
          // If this is the first instance of the node encountered (it hasn't
          // been explored yet), add it to stack so it's recursed upon
          if (arg->duplications == 0) {
            stack.push_back(arg.Get());
          }
          ++arg->duplications;
        }
      }
    }

    stack.clear();
    stack.emplace_back(variable.expr.Get());

    while (!stack.empty()) {
      auto& currentNode = stack.back();
      stack.pop_back();

      // BFS tape sorted from parent to child.
      row.emplace_back(currentNode);

      for (auto&& arg : currentNode->args) {
        // Only add node if it's not a constant and doesn't already exist in the
        // tape.
        if (arg != nullptr && arg->type != ExpressionType::kConstant) {
          // Once the number of node visitations equals the number of
          // duplications (the counter hits zero), add it to the stack. Note
          // that this means the node is only enqueued once.
          --arg->duplications;
          if (arg->duplications == 0) {
            stack.push_back(arg.Get());
          }
        }
      }
    }

    m_graph.emplace_back(std::move(row));
  }

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = row;
  }

  // Reserve triplet space for 99% sparsity
  m_cachedTriplets.reserve(m_variables.rows() * m_wrt.rows() * 0.01);

  for (int row = 0; row < m_variables.rows(); ++row) {
    if (m_variables(row).expr->type == ExpressionType::kLinear) {
      // If the row is linear, compute its gradient once here and cache its
      // triplets. Constant rows are ignored because their gradients have no
      // nonzero triplets.
      ComputeRow(row, m_cachedTriplets);
    } else if (m_variables(row).expr->type > ExpressionType::kLinear) {
      // If the row is quadratic or nonlinear, add it to the list of nonlinear
      // rows to be recomputed in Calculate().
      m_nonlinearRows.emplace_back(row);
    }
  }

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = -1;
  }

  if (m_nonlinearRows.empty()) {
    m_J.setFromTriplets(m_cachedTriplets.begin(), m_cachedTriplets.end());
  }

  m_profiler.StopSetup();
}

const Eigen::SparseMatrix<double>& Jacobian::Calculate() {
  if (m_nonlinearRows.empty()) {
    return m_J;
  }

  m_profiler.StartSolve();

  Update();

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = row;
  }

  // Copy the cached triplets so triplets added for the nonlinear rows are
  // thrown away at the end of the function
  auto triplets = m_cachedTriplets;

  for (int row : m_nonlinearRows) {
    ComputeRow(row, triplets);
  }

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = -1;
  }

  m_J.setFromTriplets(triplets.begin(), triplets.end());

  m_profiler.StopSolve();

  return m_J;
}

void Jacobian::Update() {
  for (size_t row = 0; row < m_graph.size(); ++row) {
    for (int col = m_graph[row].size() - 1; col >= 0; --col) {
      auto& node = m_graph[row][col];

      auto& lhs = node->args[0];
      auto& rhs = node->args[1];

      if (lhs != nullptr) {
        if (rhs != nullptr) {
          node->value = node->valueFunc(lhs->value, rhs->value);
        } else {
          node->value = node->valueFunc(lhs->value, 0.0);
        }
      }
    }
  }
}

Profiler& Jacobian::GetProfiler() {
  return m_profiler;
}

void Jacobian::ComputeRow(int rowIndex,
                          std::vector<Eigen::Triplet<double>>& triplets) {
  auto& row = m_graph[rowIndex];

  for (auto col : row) {
    col->adjoint = 0.0;
  }
  row[0]->adjoint = 1.0;

  for (auto col : row) {
    auto& lhs = col->args[0];
    auto& rhs = col->args[1];

    if (lhs != nullptr) {
      if (rhs != nullptr) {
        lhs->adjoint +=
            col->gradientValueFuncs[0](lhs->value, rhs->value, col->adjoint);
        rhs->adjoint +=
            col->gradientValueFuncs[1](lhs->value, rhs->value, col->adjoint);
      } else {
        lhs->adjoint +=
            col->gradientValueFuncs[0](lhs->value, 0.0, col->adjoint);
      }
    }

    if (col->row != -1) {
      triplets.emplace_back(rowIndex, col->row, col->adjoint);
    }
  }
}
