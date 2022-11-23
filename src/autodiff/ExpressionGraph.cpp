// Copyright (c) Joshua Nichols and Tyler Veness

#include "sleipnir/autodiff/ExpressionGraph.hpp"

using namespace sleipnir::autodiff;

ExpressionGraph::ExpressionGraph(Expression& root) {
  // BFS list sorted from parent to child.
  std::vector<Expression*> stack;

  stack.emplace_back(&root);

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

  stack.emplace_back(&root);

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
  //
  // Breadth-first search (BFS) is used as opposed to a depth-first search (DFS)
  // to avoid counting duplicate nodes multiple times. A list of nodes ordered
  // from parent to child with no duplicates is generated.
  //
  // https://en.wikipedia.org/wiki/Breadth-first_search
  for (int col = m_list.size() - 1; col >= 0; --col) {
    auto& node = m_list[col];

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

const std::vector<Expression*>& ExpressionGraph::GetList() const {
  return m_list;
}
