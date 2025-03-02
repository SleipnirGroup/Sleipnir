// Copyright (c) Sleipnir contributors

#pragma once

#include <functional>
#include <span>

#include <Eigen/Core>

#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/optimization/solver_config.hpp"
#include "sleipnir/optimization/solver_iteration_info.hpp"
#include "sleipnir/optimization/solver_status.hpp"
#include "sleipnir/util/symbol_exports.hpp"

namespace slp {

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

@param[in] decision_variables The list of decision variables.
@param[in] equality_constraints The list of equality constraints.
@param[in] inequality_constraints The list of inequality constraints.
@param[in] f The cost function.
@param[in] callbacks The list of user callbacks.
@param[in] config Configuration options for the solver.
@param[in,out] x The initial guess and output location for the decision
  variables.
@param[out] status The solver status.
*/
SLEIPNIR_DLLEXPORT void interior_point(
    std::span<Variable> decision_variables,
    std::span<Variable> equality_constraints,
    std::span<Variable> inequality_constraints, Variable& f,
    std::span<std::function<bool(const SolverIterationInfo& info)>> callbacks,
    const SolverConfig& config, Eigen::VectorXd& x, SolverStatus* status);

}  // namespace slp
