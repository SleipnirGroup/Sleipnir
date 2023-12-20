// Copyright (c) Sleipnir contributors

#pragma once

#include <vector>

#include <Eigen/SparseCore>

#include "sleipnir/autodiff/Expression.hpp"
#include "sleipnir/autodiff/ExpressionGraph.hpp"
#include "sleipnir/autodiff/Profiler.hpp"
#include "sleipnir/autodiff/Variable.hpp"
#include "sleipnir/autodiff/VariableMatrix.hpp"
#include "sleipnir/util/SymbolExports.hpp"

namespace sleipnir {

/**
 * This class calculates the Jacobian of a vector of variables with respect to a
 * vector of variables.
 *
 * The Jacobian is only recomputed if the variable expression is quadratic or
 * higher order.
 */
class SLEIPNIR_DLLEXPORT Jacobian {
 public:
  /**
   * Constructs a Jacobian object.
   *
   * @param variables Vector of variables of which to compute the Jacobian.
   * @param wrt Vector of variables with respect to which to compute the
   *   Jacobian.
   */
  Jacobian(const VariableMatrix& variables, const VariableMatrix& wrt) noexcept;

  /**
   * Calculates the Jacobian.
   */
  const Eigen::SparseMatrix<double>& Calculate();

  /**
   * Updates the values of the variables.
   */
  void Update();

  /**
   * Returns the profiler.
   */
  Profiler& GetProfiler();

 private:
  VariableMatrix m_variables;
  VariableMatrix m_wrt;

  std::vector<detail::ExpressionGraph> m_graphs;

  Eigen::SparseMatrix<double> m_J{m_variables.Rows(), m_wrt.Rows()};

  // Cached triplets for gradients of linear rows
  std::vector<Eigen::Triplet<double>> m_cachedTriplets;

  // List of row indices for nonlinear rows whose graients will be computed in
  // Calculate()
  std::vector<int> m_nonlinearRows;

  Profiler m_profiler;
};

}  // namespace sleipnir
