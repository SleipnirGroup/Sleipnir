// Copyright (c) Sleipnir contributors

#pragma once

#include <vector>

#include <Eigen/SparseCore>

#include "sleipnir/autodiff/Jacobian.hpp"
#include "sleipnir/autodiff/Profiler.hpp"
#include "sleipnir/autodiff/Variable.hpp"
#include "sleipnir/util/SymbolExports.hpp"

namespace sleipnir {

/**
 * This class calculates the gradient of a a variable with respect to a vector
 * of variables.
 *
 * The gradient is only recomputed if the variable expression is quadratic or
 * higher order.
 */
class SLEIPNIR_DLLEXPORT Gradient {
 public:
  /**
   * Constructs a Gradient object.
   *
   * @param variable Variable of which to compute the gradient.
   * @param wrt Variable with respect to which to compute the gradient.
   */
  Gradient(Variable variable, Variable wrt) noexcept;

  /**
   * Constructs a Gradient object.
   *
   * @param variable Variable of which to compute the gradient.
   * @param wrt Vector of variables with respect to which to compute the
   *   gradient.
   */
  Gradient(Variable variable, const VariableMatrix& wrt) noexcept;

  /**
   * Calculates the gradient.
   */
  const Eigen::SparseVector<double>& Calculate();

  /**
   * Updates the value of the variable.
   */
  void Update();

  /**
   * Returns the profiler.
   */
  Profiler& GetProfiler();

 private:
  Eigen::SparseVector<double> m_g;

  Jacobian m_jacobian;
};

}  // namespace sleipnir
