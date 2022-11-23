// Copyright (c) Joshua Nichols and Tyler Veness

#pragma once

#include <vector>

#include "sleipnir/SymbolExports.hpp"
#include "sleipnir/autodiff/Expression.hpp"
#include "sleipnir/autodiff/Variable.hpp"

namespace sleipnir::autodiff {

/**
 * This class is an adaptor type that performs value updates of an expression's
 * computational graph in a way that skips duplicates.
 */
class SLEIPNIR_DLLEXPORT ExpressionGraph {
 public:
  /**
   * Generates the deduplicated computational graph for the given expression.
   *
   * @param root The root node of the expression.
   */
  explicit ExpressionGraph(Variable& root);

  /**
   * Update the values of all nodes in this computational tree based on the
   * values of their dependent nodes.
   */
  void Update();

  /**
   * Updates the adjoints in the expression graph, effectively computing the
   * gradient.
   *
   * @param func A function that takes two arguments: an int for the gradient
   *   row, and a double for the adjoint (gradient value).
   */
  template <typename F>
  void ComputeAdjoints(F&& func) {
    // Zero adjoints. The root node's adjoint is 1.0 as df/df is always 1.
    for (auto col : m_list) {
      col->adjoint = 0.0;
    }
    m_list[0]->adjoint = 1.0;

    // df/dx = (df/dy)(dy/dx). The adjoint of x is equal to the adjoint of y
    // multiplied by dy/dx. If there are multiple "paths" from the root node to
    // variable; the variable's adjoint is the sum of each path's adjoint
    // contribution.
    for (auto col : m_list) {
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
        func(col->row, col->adjoint);
      }
    }
  }

 private:
  std::vector<Expression*> m_list;
};

}  // namespace sleipnir::autodiff
