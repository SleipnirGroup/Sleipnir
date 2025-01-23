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
Finds the optimal solution to a nonlinear program using Sequential Quadratic
Programming (SQP).

A nonlinear program has the form:

@verbatim
     min_x f(x)
subject to cₑ(x) = 0
@endverbatim

where f(x) is the cost function and cₑ(x) are the equality constraints.

@param[in] decisionVariables The list of decision variables.
@param[in] equalityConstraints The list of equality constraints.
@param[in] f The cost function.
@param[in] callbacks The list of user callbacks.
@param[in] config Configuration options for the solver.
@param[in,out] x The initial guess and output location for the decision
  variables.
@param[out] status The solver status.
*/
SLEIPNIR_DLLEXPORT void SQP(
    std::span<Variable> decisionVariables,
    std::span<Variable> equalityConstraints, Variable& f,
    std::span<std::function<bool(const SolverIterationInfo& info)>> callbacks,
    const SolverConfig& config, Eigen::VectorXd& x, SolverStatus* status);

}  // namespace sleipnir
