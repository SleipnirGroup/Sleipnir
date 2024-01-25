// Copyright (c) Sleipnir contributors

#include <pybind11/pybind11.h>

#include "autodiff/BindExpressionType.hpp"
#include "autodiff/BindVariable.hpp"
#include "autodiff/BindVariableMatrices.hpp"
#include "optimization/BindConstraints.hpp"
#include "optimization/BindOptimizationProblem.hpp"
#include "optimization/BindSolverExitCondition.hpp"
#include "optimization/BindSolverIterationInfo.hpp"
#include "optimization/BindSolverStatus.hpp"

namespace py = pybind11;

namespace sleipnir {

PYBIND11_MODULE(_jormungandr, m) {
  using namespace sleipnir;

  m.doc() =
      "A linearity-exploiting sparse nonlinear constrained optimization "
      "problem solver that uses the interior-point method.";

  py::module_ autodiff = m.def_submodule("autodiff");
  py::module_ optimization = m.def_submodule("optimization");

  BindExpressionType(autodiff);
  BindConstraints(optimization);
  BindVariable(autodiff);
  BindVariableMatrices(autodiff);

  BindSolverExitCondition(optimization);
  BindSolverIterationInfo(optimization);
  BindSolverStatus(optimization);
  BindOptimizationProblem(optimization);
}

}  // namespace sleipnir
