// Copyright (c) Joshua Nichols and Tyler Veness

#include "sleipnir/autodiff/Gradient.hpp"

#include <tuple>

#include <wpi/DenseMap.h>

#include "sleipnir/IntrusiveSharedPtr.hpp"

using namespace sleipnir::autodiff;

Gradient::Gradient(Variable variable, Variable wrt) noexcept
    : Gradient{std::move(variable), MapVectorXvar{&wrt, 1}} {}

Gradient::Gradient(Variable variable, Eigen::Ref<VectorXvar> wrt) noexcept
    : m_variable{std::move(variable)}, m_wrt{wrt}, m_g{m_wrt.rows()} {
  if (m_variable.Type() < ExpressionType::kLinear) {
    // If the expression is less than linear, the Jacobian is zero
    m_profiler.StartSolve();
    m_g.setZero();
    m_profiler.StopSolve();
  } else {
    // BFS
    std::vector<Expression*> stack;

    m_graph.clear();

    stack.emplace_back(m_variable.expr.Get());

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
    stack.emplace_back(m_variable.expr.Get());

    while (!stack.empty()) {
      auto& currentNode = stack.back();
      stack.pop_back();

      // BFS tape sorted from parent to child.
      m_graph.emplace_back(currentNode);

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

    if (m_variable.expr->type == ExpressionType::kLinear) {
      // If the expression is linear, compute its gradient once here and cache
      // its value. Constant expressions are ignored because their gradients
      // have no nonzero values.
      Compute();
    }
  }
}

const Eigen::SparseVector<double>& Gradient::Calculate() {
  if (m_variable.Type() > ExpressionType::kLinear) {
    Compute();
  }

  return m_g;
}

void Gradient::Update() {
  for (int col = m_graph.size() - 1; col >= 0; --col) {
    auto& node = m_graph[col];

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

Profiler& Gradient::GetProfiler() {
  return m_profiler;
}

void Gradient::Compute() {
  Update();

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = row;
  }

  for (auto col : m_graph) {
    col->adjoint = 0.0;
  }
  m_graph[0]->adjoint = 1.0;

  for (auto col : m_graph) {
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
      m_g.coeffRef(col->row) = col->adjoint;
    }
  }

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = -1;
  }
}
