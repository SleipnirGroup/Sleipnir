// Copyright (c) Joshua Nichols and Tyler Veness

#pragma once

#include <vector>

#include "sleipnir/SymbolExports.hpp"
#include "sleipnir/autodiff/Expression.hpp"

namespace sleipnir::autodiff {

/**
 * This class is an adaptor type that performs value updates of an expression's
 * computational graph in a way that skips duplicates.
 */
class SLEIPNIR_DLLEXPORT ExpressionGraph {
 public:
  constexpr ExpressionGraph() = default;

  /**
   * Generates the deduplicated computational graph for the given expression.
   *
   * @param root The root node of the expression.
   */
  explicit ExpressionGraph(Expression& root);

  /**
   * Update the values of all nodes in this computational tree based on the
   * values of their dependent nodes.
   */
  void Update();

  /**
   * Returns the graph's list of nodes.
   */
  const std::vector<Expression*>& GetList() const;

 private:
  std::vector<Expression*> m_list;
};

}  // namespace sleipnir::autodiff
