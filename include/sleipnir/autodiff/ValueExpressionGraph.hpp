// Copyright (c) Sleipnir contributors

#pragma once

#include <ranges>

#include "sleipnir/autodiff/Expression.hpp"
#include "sleipnir/util/small_vector.hpp"

namespace sleipnir::detail {

/**
 * This class is an adaptor type that performs value updates of an expression's
 * value graph in a way that skips duplicates.
 */
class ValueExpressionGraph {
 public:
  /**
   * Generates the deduplicated value graph for the given expression.
   *
   * @param root The root node of the expression.
   */
  explicit ValueExpressionGraph(const ExpressionPtr& root) {
    // If the root type is a constant, Update() is a no-op, so there's no work
    // to do
    if (root == nullptr || root->Type() == ExpressionType::kConstant) {
      return;
    }

    // Breadth-first search (BFS) is used as opposed to a depth-first search
    // (DFS) to avoid counting duplicate nodes multiple times. A list of nodes
    // ordered from parent to child with no duplicates is generated.
    //
    // https://en.wikipedia.org/wiki/Breadth-first_search

    small_vector<Expression*> stack;

    // Assign each node's number of instances in the tree to
    // Expression::duplications
    stack.emplace_back(root.Get());
    while (!stack.empty()) {
      auto node = stack.back();
      stack.pop_back();

      for (auto& arg : node->args) {
        if (arg != nullptr) {
          // If this is the first instance of the node encountered (it hasn't
          // been explored yet), add it to stack so it's recursed upon
          if (arg->duplications == 0) {
            stack.push_back(arg.Get());
          }
          ++arg->duplications;
        }
      }
    }

    // Generate BFS lists sorted from parent to child
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
        if (arg != nullptr) {
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
