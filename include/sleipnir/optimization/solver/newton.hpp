// Copyright (c) Sleipnir contributors

#pragma once

#include <functional>
#include <span>

#include <Eigen/Core>

#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/optimization/solver/exit_status.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/util/symbol_exports.hpp"

namespace slp {

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
@param[in] options Solver options.
@param[in,out] x The initial guess and output location for the decision
  variables.
@return The exit status.
*/
SLEIPNIR_DLLEXPORT ExitStatus
newton(std::span<Variable> decision_variables, Variable& f,
       std::span<std::function<bool(const IterationInfo& info)>> callbacks,
       const Options& options, Eigen::VectorXd& x);

}  // namespace slp
