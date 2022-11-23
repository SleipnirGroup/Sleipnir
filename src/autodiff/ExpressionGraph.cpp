// Copyright (c) Joshua Nichols and Tyler Veness

#include "sleipnir/autodiff/ExpressionGraph.hpp"

using namespace sleipnir::autodiff;

ExpressionGraph::ExpressionGraph(Variable& root) {
  // If the root type is a constant, Update() is a no-op, so there's no work to
  // do
  if (root.Type() == ExpressionType::kConstant) {
    return;
  }

  // Breadth-first search (BFS) is used as opposed to a depth-first search (DFS)
  // to avoid counting duplicate nodes multiple times. A list of nodes ordered
  // from parent to child with no duplicates is generated.
  //
  // https://en.wikipedia.org/wiki/Breadth-first_search

  // BFS list sorted from parent to child.
  std::vector<Expression*> stack;

  stack.emplace_back(root.expr.Get());

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

  stack.emplace_back(root.expr.Get());

  while (!stack.empty()) {
    auto& currentNode = stack.back();
    stack.pop_back();

    // BFS list sorted from parent to child.
    m_list.emplace_back(currentNode);

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
}

void ExpressionGraph::Update() {
  // Traverse the BFS list backward from child to parent and update the value of
  // each node.
  for (auto it = m_list.rbegin(); it != m_list.rend(); ++it) {
    auto& node = *it;

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

VectorXvar ExpressionGraph::GenerateGradientTree(Eigen::Ref<VectorXvar> wrt) {
  // Read wpimath/README.md#Reverse_accumulation_automatic_differentiation for
  // background on reverse accumulation automatic differentiation.

  for (int row = 0; row < wrt.rows(); ++row) {
    wrt(row).expr->row = row;
  }

  VectorXvar grad{wrt.rows()};
  grad.fill(Variable{});

  // Zero adjoints. The root node's adjoint is 1.0 as df/df is always 1.
  for (auto col : m_list) {
    col->adjointExpr = nullptr;
  }
  m_list[0]->adjointExpr = MakeConstant(1.0);

  // df/dx = (df/dy)(dy/dx). The adjoint of x is equal to the adjoint of y
  // multiplied by dy/dx. If there are multiple "paths" from the root node to
  // variable; the variable's adjoint is the sum of each path's adjoint
  // contribution.
  for (auto col : m_list) {
    auto& lhs = col->args[0];
    auto& rhs = col->args[1];

    if (lhs != nullptr) {
      lhs->adjointExpr =
          lhs->adjointExpr + col->gradientFuncs[0](lhs, rhs, col->adjointExpr);
      if (rhs != nullptr) {
        rhs->adjointExpr = rhs->adjointExpr +
                           col->gradientFuncs[1](lhs, rhs, col->adjointExpr);
      }
    }

    // If variable is a leaf node, assign its adjoint to the gradient.
    if (col->row != -1) {
      grad(col->row) = Variable{col->adjointExpr};
    }
  }

  for (int row = 0; row < wrt.rows(); ++row) {
    wrt(row).expr->row = -1;
  }

  return grad;
}
