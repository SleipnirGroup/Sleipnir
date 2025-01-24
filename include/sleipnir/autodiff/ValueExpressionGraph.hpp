// Copyright (c) Sleipnir contributors

#pragma once

#include <ranges>

#include "sleipnir/autodiff/Expression.hpp"
#include "sleipnir/util/small_vector.hpp"

namespace sleipnir::detail {

/**
 * This class is an adaptor type that performs value updates of an expression's
 * value graph.
 */
class ValueExpressionGraph {
 public:
  /**
   * Generates the value graph for the given expression.
   *
   * @param root The root node of the expression.
   */
  explicit ValueExpressionGraph(const ExpressionPtr& root) {
    // If the root type is a constant, Update() is a no-op, so there's no work
    // to do
    if (root == nullptr || root->Type() == ExpressionType::kConstant) {
      return;
    }

    // Stack of nodes to explore
    small_vector<Expression*> stack;

    // Enumerate incoming edges for each node via depth-first search
    stack.emplace_back(root.Get());
    while (!stack.empty()) {
      auto node = stack.back();
      stack.pop_back();

      for (auto& arg : node->args) {
        // If the node hasn't been explored yet, add it to the stack
        if (arg != nullptr && ++arg->incomingEdges == 1) {
          stack.push_back(arg.Get());
        }
      }
    }

    // Generate topological sort of graph from parent to child.
    //
    // A node is only added to the stack after all its incoming edges have been
    // traversed. Expression::incomingEdges is a decrementing counter for
    // tracking this.
    //
    // https://en.wikipedia.org/wiki/Topological_sorting
    stack.emplace_back(root.Get());
    while (!stack.empty()) {
      auto node = stack.back();
      stack.pop_back();

      if (node->args[0] != nullptr) {
        // Constants (expressions with no arguments) are skipped because they
        // don't need to be updated
        m_valueList.emplace_back(node);
      }

      for (auto& arg : node->args) {
        // If we traversed all this node's incoming edges, add it to the stack
        if (arg != nullptr && --arg->incomingEdges == 0) {
          stack.push_back(arg.Get());
        }
      }
    }
  }

  /**
   * Update the values of all nodes in this value graph based on the values of
   * their dependent nodes.
   */
  void Update() {
    // Traverse the BFS list backward from child to parent and update the value
    // of each node.
    for (auto& node : m_valueList | std::views::reverse) {
      auto& lhs = node->args[0];
      auto& rhs = node->args[1];

      if (lhs != nullptr) {
        if (rhs != nullptr) {
          node->value = node->Value(lhs->value, rhs->value);
        } else {
          node->value = node->Value(lhs->value, 0.0);
        }
      }
    }
  }

 private:
  // List for updating values
  small_vector<Expression*> m_valueList;
};

}  // namespace sleipnir::detail
