// Copyright (c) Sleipnir contributors

#pragma once

#include <functional>
#include <span>

#include <Eigen/Core>

#include "sleipnir/autodiff/Variable.hpp"
#include "sleipnir/optimization/Constraints.hpp"
#include "sleipnir/optimization/SolverConfig.hpp"
#include "sleipnir/optimization/SolverIterationInfo.hpp"
#include "sleipnir/optimization/SolverStatus.hpp"

namespace sleipnir {

/**
Finds the optimal solution to a nonlinear program using the interior-point
method.

A nonlinear program has the form:

@verbatim
     min_x f(x)
subject to cₑ(x) = 0
           cᵢ(x) ≥ 0
@endverbatim

where f(x) is the cost function, cₑ(x) are the equality constraints, and cᵢ(x)
are the inequality constraints.

@param[in] decisionVariables The list of decision variables.
@param[in] equalityConstraints The list of equality constraints.
@param[in] inequalityConstraints The list of inequality constraints.
@param[in] f The cost function.
@param[in] callback The user callback.
@param[in] config Configuration options for the solver.
@param[in] feasibilityRestoration Whether to use feasibility restoration instead
  of the normal algorithm.
@param[in,out] x The initial guess and output location for the decision
  variables.
@param[in,out] s The initial guess and output location for the inequality
  constraint slack variables.
@param[out] status The solver status.
*/
void InteriorPoint(
    std::span<Variable> decisionVariables,
    std::span<Variable> equalityConstraints,
    std::span<Variable> inequalityConstraints, Variable& f,
    const std::function<bool(const SolverIterationInfo&)>& callback,
    const SolverConfig& config, bool feasibilityRestoration, Eigen::VectorXd& x,
    Eigen::VectorXd& s, SolverStatus* status);

}  // namespace sleipnir
