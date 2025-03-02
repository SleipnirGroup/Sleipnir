// Copyright (c) Sleipnir contributors

#pragma once

#include <nanobind/nanobind.h>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/autodiff/gradient.hpp>
#include <sleipnir/autodiff/hessian.hpp>
#include <sleipnir/autodiff/jacobian.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_block.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/control/ocp_solver.hpp>
#include <sleipnir/optimization/optimization_problem.hpp>
#include <sleipnir/optimization/solver_exit_condition.hpp>
#include <sleipnir/optimization/solver_iteration_info.hpp>
#include <sleipnir/optimization/solver_status.hpp>

namespace nb = nanobind;

namespace slp {

void bind_expression_type(nb::enum_<ExpressionType>& e);

void bind_variable(nb::module_& autodiff, nb::class_<Variable>& cls);
void bind_variable_matrix(nb::module_& autodiff,
                          nb::class_<VariableMatrix>& cls);
void bind_variable_block(nb::class_<VariableBlock<VariableMatrix>>& cls);

void bind_gradient(nb::class_<Gradient>& cls);
void bind_hessian(nb::class_<Hessian<>>& cls);
void bind_jacobian(nb::class_<Jacobian>& cls);

void bind_equality_constraints(nb::class_<EqualityConstraints>& cls);
void bind_inequality_constraints(nb::class_<InequalityConstraints>& cls);

void bind_solver_exit_condition(nb::enum_<SolverExitCondition>& e);
void bind_solver_iteration_info(nb::class_<SolverIterationInfo>& cls);
void bind_solver_status(nb::class_<SolverStatus>& cls);

void bind_optimization_problem(nb::class_<OptimizationProblem>& cls);

void bind_ocp_solver(nb::enum_<TranscriptionMethod>& transcription_method,
                     nb::enum_<DynamicsType>& dynamics_type,
                     nb::enum_<TimestepMethod>& timestep_method,
                     nb::class_<OCPSolver, OptimizationProblem>& cls);

}  // namespace slp
