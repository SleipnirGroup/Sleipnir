// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/SparseCore>

#include "sleipnir/autodiff/adjoint_expression_graph.hpp"
#include "sleipnir/autodiff/jacobian.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"
#include "sleipnir/util/small_vector.hpp"
#include "sleipnir/util/solve_profiler.hpp"
#include "sleipnir/util/symbol_exports.hpp"

namespace sleipnir {

/**
 * This class calculates the Hessian of a variable with respect to a vector of
 * variables.
 *
 * The gradient tree is cached so subsequent Hessian calculations are faster,
 * and the Hessian is only recomputed if the variable expression is nonlinear.
 */
class SLEIPNIR_DLLEXPORT Hessian {
 public:
  /**
   * Constructs a Hessian object.
   *
   * @param variable Variable of which to compute the Hessian.
   * @param wrt Vector of variables with respect to which to compute the
   *   Hessian.
   */
  Hessian(Variable variable, VariableMatrix wrt) noexcept
      : m_jacobian{
            detail::AdjointExpressionGraph{variable}.generate_gradient_tree(
                wrt),
            wrt} {}

  /**
   * Returns the Hessian as a VariableMatrix.
   *
   * This is useful when constructing optimization problems with derivatives in
   * them.
   *
   * @return The Hessian as a VariableMatrix.
   */
  VariableMatrix get() const { return m_jacobian.get(); }

  /**
   * Evaluates the Hessian at wrt's value.
   *
   * @return The Hessian at wrt's value.
   */
  const Eigen::SparseMatrix<double>& value() { return m_jacobian.value(); }

  /**
   * Returns the profilers.
   *
   * @return The profilers.
   */
  const small_vector<SolveProfiler>& get_profilers() const {
    return m_jacobian.get_profilers();
  }

 private:
  Jacobian m_jacobian;
};

}  // namespace sleipnir
