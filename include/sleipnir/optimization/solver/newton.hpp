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

namespace sleipnir {

/**
Finds the optimal solution to a nonlinear program using Newton's method.

A nonlinear program has the form:

@verbatim
     min_x f(x)
@endverbatim

where f(x) is the cost function.

@param[in] decision_variables The list of decision variables.
@param[in] f The cost function.
@param[in] callbacks The list of user callbacks.
@param[in] config Configuration options for the solver.
@param[in,out] x The initial guess and output location for the decision
  variables.
@param[out] status The solver status.
*/
SLEIPNIR_DLLEXPORT void newton(
    std::span<Variable> decision_variables, Variable& f,
    std::span<std::function<bool(const SolverIterationInfo& info)>> callbacks,
    const SolverConfig& config, Eigen::VectorXd& x, SolverStatus* status);

}  // namespace sleipnir
