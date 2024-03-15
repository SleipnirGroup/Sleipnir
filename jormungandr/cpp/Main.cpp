// Copyright (c) Sleipnir contributors

#include <pybind11/pybind11.h>

#include "Binders.hpp"

namespace py = pybind11;

namespace sleipnir {

PYBIND11_MODULE(_jormungandr, m) {
  m.doc() =
      "A linearity-exploiting sparse nonlinear constrained optimization "
      "problem solver that uses the interior-point method.";

  py::module_ autodiff = m.def_submodule("autodiff");
  BindExpressionType(autodiff);
  BindGradient(autodiff);
  BindHessian(autodiff);
  BindJacobian(autodiff);
  BindVariable(autodiff);
  BindVariableMatrix(autodiff);
  BindVariableBlock(autodiff);  // Must be bound after VariableMatrix

  py::module_ optimization = m.def_submodule("optimization");
  BindConstraints(optimization);
  BindSolverExitCondition(optimization);
  BindSolverIterationInfo(optimization);
  BindSolverStatus(optimization);
  BindOptimizationProblem(optimization);
}

}  // namespace sleipnir
