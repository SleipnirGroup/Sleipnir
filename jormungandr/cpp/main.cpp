// Copyright (c) Sleipnir contributors

#include <nanobind/nanobind.h>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_block.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/control/ocp_solver.hpp>
#include <sleipnir/optimization/optimization_problem.hpp>
#include <sleipnir/optimization/solver_exit_condition.hpp>
#include <sleipnir/optimization/solver_iteration_info.hpp>
#include <sleipnir/optimization/solver_status.hpp>

#include "binders.hpp"
#include "docstrings.hpp"

namespace nb = nanobind;

namespace sleipnir {

NB_MODULE(_jormungandr, m) {
  m.doc() =
      "A linearity-exploiting sparse nonlinear constrained optimization "
      "problem solver that uses the interior-point method.";

  nb::module_ autodiff = m.def_submodule("autodiff");
  nb::module_ optimization = m.def_submodule("optimization");
  nb::module_ control = m.def_submodule("control");

  nb::enum_<ExpressionType> expression_type{autodiff, "ExpressionType",
                                            DOC(sleipnir, ExpressionType)};

  nb::class_<Variable> variable{autodiff, "Variable", DOC(sleipnir, Variable)};
  nb::class_<VariableMatrix> variable_matrix{autodiff, "VariableMatrix",
                                             DOC(sleipnir, VariableMatrix)};
  nb::class_<VariableBlock<VariableMatrix>> variable_block{
      autodiff, "VariableBlock", DOC(sleipnir, VariableBlock)};

  nb::class_<Gradient> gradient{autodiff, "Gradient", DOC(sleipnir, Gradient)};
  nb::class_<Hessian> hessian{autodiff, "Hessian", DOC(sleipnir, Hessian)};
  nb::class_<Jacobian> jacobian{autodiff, "Jacobian", DOC(sleipnir, Jacobian)};

  nb::class_<EqualityConstraints> equality_constraints{
      optimization, "EqualityConstraints", DOC(sleipnir, EqualityConstraints)};
  nb::class_<InequalityConstraints> inequality_constraints{
      optimization, "InequalityConstraints",
      DOC(sleipnir_InequalityConstraints)};

  nb::enum_<SolverExitCondition> solver_exit_condition{
      optimization, "SolverExitCondition", DOC(sleipnir, SolverExitCondition)};
  nb::class_<SolverIterationInfo> solver_iteration_info{
      optimization, "SolverIterationInfo", DOC(sleipnir, SolverIterationInfo)};
  nb::class_<SolverStatus> solver_status{optimization, "SolverStatus",
                                         DOC(sleipnir, SolverStatus)};

  nb::class_<OptimizationProblem> optimization_problem{
      optimization, "OptimizationProblem", DOC(sleipnir, OptimizationProblem)};

  nb::enum_<TranscriptionMethod> transcription_method{
      control, "TranscriptionMethod", DOC(sleipnir, TranscriptionMethod)};
  nb::enum_<DynamicsType> dynamics_type{control, "DynamicsType",
                                        DOC(sleipnir, DynamicsType)};
  nb::enum_<TimestepMethod> timestep_method{control, "TimestepMethod",
                                            DOC(sleipnir, TimestepMethod)};
  nb::class_<OCPSolver, OptimizationProblem> ocp_solver{
      control, "OCPSolver", DOC(sleipnir, OCPSolver)};

  bind_expression_type(expression_type);

  bind_variable(autodiff, variable);
  bind_variable_matrix(autodiff, variable_matrix);
  bind_variable_block(variable_block);

  // Implicit conversions
  variable.def(nb::init_implicit<VariableMatrix>());
  variable_matrix.def(nb::init_implicit<VariableBlock<VariableMatrix>>());

  bind_gradient(gradient);
  bind_hessian(hessian);
  bind_jacobian(jacobian);

  bind_equality_constraints(equality_constraints);
  bind_inequality_constraints(inequality_constraints);

  bind_solver_exit_condition(solver_exit_condition);
  bind_solver_iteration_info(solver_iteration_info);
  bind_solver_status(solver_status);

  bind_optimization_problem(optimization_problem);

  bind_ocp_solver(transcription_method, dynamics_type, timestep_method,
                  ocp_solver);
}

}  // namespace sleipnir
