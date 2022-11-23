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
   * Returns a reference to the given element in the BFS list.
   */
  const Expression* operator[](size_t index) const { return m_list[index]; }

  /**
   * Returns a reference to the given element in the BFS list.
   */
  Expression* operator[](size_t index) { return m_list[index]; }

  /**
   * Returns an iterator to the beginning of the BFS list.
   */
  decltype(auto) begin() { return m_list.begin(); }

  /**
   * Returns an iterator to the end of the BFS list.
   */
  decltype(auto) end() { return m_list.end(); }

 private:
  std::vector<Expression*> m_list;
};

}  // namespace sleipnir::autodiff
