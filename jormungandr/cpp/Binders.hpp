// Copyright (c) Sleipnir contributors

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace sleipnir {

void BindExpressionType(py::module_& autodiff);
void BindGradient(py::module_& autodiff);
void BindHessian(py::module_& autodiff);
void BindJacobian(py::module_& autodiff);
void BindVariable(py::module_& autodiff);
void BindVariableBlock(py::module_& autodiff);
void BindVariableMatrix(py::module_& autodiff);

void BindConstraints(py::module_& optimization);
void BindOptimizationProblem(py::module_& optimization);
void BindSolverExitCondition(py::module_& optimization);
void BindSolverIterationInfo(py::module_& optimization);
void BindSolverStatus(py::module_& optimization);

}  // namespace sleipnir
