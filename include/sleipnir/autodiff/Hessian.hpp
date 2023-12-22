// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "sleipnir/autodiff/Jacobian.hpp"
#include "sleipnir/autodiff/Profiler.hpp"
#include "sleipnir/autodiff/Variable.hpp"
#include "sleipnir/util/SymbolExports.hpp"

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
  Hessian(Variable variable, const VariableMatrix& wrt) noexcept;

  /**
   * Calculates the Hessian.
   */
  const Eigen::SparseMatrix<double>& Calculate();

  /**
   * Updates the values of the gradient tree.
   */
  void Update();

  /**
   * Returns the profiler.
   */
  Profiler& GetProfiler();

 private:
  Jacobian m_jacobian;
};

}  // namespace sleipnir
