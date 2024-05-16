// Copyright (c) Sleipnir contributors

#include <pybind11/pybind11.h>
#include <sleipnir/autodiff/Variable.hpp>
#include <sleipnir/autodiff/VariableBlock.hpp>
#include <sleipnir/autodiff/VariableMatrix.hpp>
#include <sleipnir/optimization/Constraints.hpp>
#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <sleipnir/optimization/SolverExitCondition.hpp>
#include <sleipnir/optimization/SolverIterationInfo.hpp>
#include <sleipnir/optimization/SolverStatus.hpp>

#include "Binders.hpp"
#include "Docstrings.hpp"

namespace py = pybind11;

namespace sleipnir {

PYBIND11_MODULE(_jormungandr, m) {
  m.doc() =
      "A linearity-exploiting sparse nonlinear constrained optimization "
      "problem solver that uses the interior-point method.";

  py::module_ autodiff = m.def_submodule("autodiff");
  py::module_ optimization = m.def_submodule("optimization");

  py::enum_<ExpressionType> expression_type{autodiff, "ExpressionType",
                                            DOC(sleipnir, ExpressionType)};

  py::class_<Variable> variable{autodiff, "Variable", DOC(sleipnir, Variable)};
  py::class_<VariableMatrix> variable_matrix{autodiff, "VariableMatrix",
                                             DOC(sleipnir, VariableMatrix)};
  py::class_<VariableBlock<VariableMatrix>> variable_block{
      autodiff, "VariableBlock", DOC(sleipnir, VariableBlock)};

  py::class_<Gradient> gradient{autodiff, "Gradient", DOC(sleipnir, Gradient)};
  py::class_<Hessian> hessian{autodiff, "Hessian", DOC(sleipnir, Hessian)};
  py::class_<Jacobian> jacobian{autodiff, "Jacobian", DOC(sleipnir, Jacobian)};

  py::class_<EqualityConstraints> equality_constraints{
      optimization, "EqualityConstraints", DOC(sleipnir, EqualityConstraints)};
  py::class_<InequalityConstraints> inequality_constraints{
      optimization, "InequalityConstraints",
      DOC(sleipnir_InequalityConstraints)};

  py::enum_<SolverExitCondition> solver_exit_condition{
      optimization, "SolverExitCondition", DOC(sleipnir, SolverExitCondition)};
  py::class_<SolverIterationInfo> solver_iteration_info{
      optimization, "SolverIterationInfo", DOC(sleipnir, SolverIterationInfo)};
  py::class_<SolverStatus> solver_status{optimization, "SolverStatus",
                                         DOC(sleipnir, SolverStatus)};

  py::class_<OptimizationProblem> optimization_problem{
      optimization, "OptimizationProblem", DOC(sleipnir, OptimizationProblem)};

  BindExpressionType(expression_type);

  BindVariable(autodiff, variable);
  BindVariableMatrix(autodiff, variable_matrix);
  BindVariableBlock(autodiff, variable_block);

  BindGradient(gradient);
  BindHessian(hessian);
  BindJacobian(jacobian);

  BindEqualityConstraints(equality_constraints);
  BindInequalityConstraints(inequality_constraints);

  BindSolverExitCondition(solver_exit_condition);
  BindSolverIterationInfo(solver_iteration_info);
  BindSolverStatus(solver_status);

  BindOptimizationProblem(optimization_problem);
}

}  // namespace sleipnir
