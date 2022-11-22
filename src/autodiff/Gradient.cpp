// Copyright (c) Joshua Nichols and Tyler Veness

#include "sleipnir/autodiff/Gradient.hpp"

#include "sleipnir/IntrusiveSharedPtr.hpp"

using namespace sleipnir::autodiff;

Gradient::Gradient(Variable variable, Variable wrt) noexcept
    : Gradient{std::move(variable), MapVectorXvar{&wrt, 1}} {}

Gradient::Gradient(Variable variable, Eigen::Ref<VectorXvar> wrt) noexcept
    : m_variable{std::move(variable)}, m_wrt{wrt}, m_g{m_wrt.rows()} {
  m_profiler.StartSetup();
  if (m_variable.Type() == ExpressionType::kConstant) {
    // If the expression is constant, the gradient is zero.
    m_g.setZero();
  } else {
    // Breadth-first search (BFS) of the expression's computational tree. BFS is used as
    // opposed to a depth-first search (DFS) to avoid counting duplicate nodes multiple times.
    // A list of nodes ordered from parent to child with no duplicates is generated for later use.
    // https://en.wikipedia.org/wiki/Breadth-first_search
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

    stack.emplace_back(m_variable.expr.Get());

    while (!stack.empty()) {
      auto& currentNode = stack.back();
      stack.pop_back();

      // BFS list sorted from parent to child.
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
      // its value.
      Compute();
    }
  }
  m_profiler.StopSetup();
}

const Eigen::SparseVector<double>& Gradient::Calculate() {
  if (m_variable.Type() > ExpressionType::kLinear) {
    m_profiler.StartSolve();
    Compute();
    m_profiler.StopSolve();
  }

  return m_g;
}

void Gradient::Update() {
  // Traverse the BFS list backward from child to parent and update the value of each node.
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
  // Computes the gradient of the expression. Given the expression f and variable x,
  // the derivative df/dx is denoted the "adjoint" of x.
  Update();

  // Assigns leaf nodes their respective position in the gradient.
  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = row;
  }

  // Zero adjoints. The root node's adjoint is 1.0 as df/df is always 1.
  for (auto col : m_graph) {
    col->adjoint = 0.0;
  }
  m_graph[0]->adjoint = 1.0;

  // df/dx = (df/dy)(dy/dx). The adjoint of x is equal to the adjoint of y multiplied by dy/dx.
  // If there are multiple "paths" from the root node to variable; the variable's adjoint is the
  // sum of each path's adjoint contribution.
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

    // If variable is a leaf node, assign its adjoint to the gradient.
    if (col->row != -1) {
      m_g.coeffRef(col->row) = col->adjoint;
    }
  }

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = -1;
  }
}
