// Copyright (c) Sleipnir contributors

#pragma once

#include <functional>
#include <span>

#include <Eigen/Core>

#include "sleipnir/autodiff/Variable.hpp"
#include "sleipnir/optimization/SolverConfig.hpp"
#include "sleipnir/optimization/SolverIterationInfo.hpp"
#include "sleipnir/optimization/SolverStatus.hpp"
#include "sleipnir/util/SymbolExports.hpp"

namespace sleipnir {

/**
Finds the optimal solution to a nonlinear program using Newton's method.

A nonlinear program has the form:

@verbatim
     min_x f(x)
@endverbatim

where f(x) is the cost function.

@param[in] decisionVariables The list of decision variables.
@param[in] f The cost function.
@param[in] callbacks The list of user callbacks.
@param[in] config Configuration options for the solver.
@param[in,out] x The initial guess and output location for the decision
  variables.
@param[out] status The solver status.
*/
SLEIPNIR_DLLEXPORT void Newton(
    std::span<Variable> decisionVariables, Variable& f,
    std::span<std::function<bool(const SolverIterationInfo& info)>> callbacks,
    const SolverConfig& config, Eigen::VectorXd& x, SolverStatus* status);

}  // namespace sleipnir
