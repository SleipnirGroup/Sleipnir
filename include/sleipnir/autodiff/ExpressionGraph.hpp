// Copyright (c) Sleipnir contributors

#pragma once

#include <concepts>
#include <vector>

#include "sleipnir/autodiff/Expression.hpp"
#include "sleipnir/autodiff/Variable.hpp"
#include "sleipnir/autodiff/VariableMatrix.hpp"
#include "sleipnir/util/SymbolExports.hpp"

namespace sleipnir::detail {

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
   * Returns the variable's gradient tree.
   *
   * @param wrt Variables with respect to which to compute the gradient.
   */
  VariableMatrix GenerateGradientTree(const VariableMatrix& wrt);

  /**
   * Updates the adjoints in the expression graph, effectively computing the
   * gradient.
   *
   * @param func A function that takes two arguments: an int for the gradient
   *   row, and a double for the adjoint (gradient value).
   */
  template <std::invocable<int, double> F>
  void ComputeAdjoints(F&& func) {
    // Zero adjoints. The root node's adjoint is 1.0 as df/df is always 1.
    m_adjointList[0]->adjoint = 1.0;
    for (auto it = m_adjointList.begin() + 1; it != m_adjointList.end(); ++it) {
      auto& node = *it;
      node->adjoint = 0.0;
    }

    // df/dx = (df/dy)(dy/dx). The adjoint of x is equal to the adjoint of y
    // multiplied by dy/dx. If there are multiple "paths" from the root node to
    // variable; the variable's adjoint is the sum of each path's adjoint
    // contribution.
    for (size_t col = 0; col < m_adjointList.size(); ++col) {
      auto& node = m_adjointList[col];
      auto& lhs = node->args[0];
      auto& rhs = node->args[1];

      lhs->adjoint +=
          node->gradientValueFuncs[0](lhs->value, rhs->value, node->adjoint);
      rhs->adjoint +=
          node->gradientValueFuncs[1](lhs->value, rhs->value, node->adjoint);

      // If variable is a leaf node, assign its adjoint to the gradient.
      int row = m_rowList[col];
      if (row != -1) {
        func(row, node->adjoint);
      }
    }
  }

 private:
  // List that maps nodes to their respective row.
  std::vector<int> m_rowList;

  // List for updating adjoints
  std::vector<Expression*> m_adjointList;

  // List for updating values
  std::vector<Expression*> m_valueList;
};

}  // namespace sleipnir::detail
