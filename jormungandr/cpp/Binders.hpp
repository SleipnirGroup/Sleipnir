// Copyright (c) Sleipnir contributors

#pragma once

#include <pybind11/pybind11.h>
#include <sleipnir/autodiff/ExpressionType.hpp>
#include <sleipnir/autodiff/Gradient.hpp>
#include <sleipnir/autodiff/Hessian.hpp>
#include <sleipnir/autodiff/Jacobian.hpp>
#include <sleipnir/autodiff/Variable.hpp>
#include <sleipnir/autodiff/VariableBlock.hpp>
#include <sleipnir/autodiff/VariableMatrix.hpp>
#include <sleipnir/optimization/Constraints.hpp>
#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <sleipnir/optimization/SolverExitCondition.hpp>
#include <sleipnir/optimization/SolverIterationInfo.hpp>
#include <sleipnir/optimization/SolverStatus.hpp>

namespace py = pybind11;

namespace sleipnir {

void BindExpressionType(py::enum_<ExpressionType>& e);

void BindVariable(py::module_& autodiff, py::class_<Variable>& cls);
void BindVariableMatrix(py::module_& autodiff, py::class_<VariableMatrix>& cls);
void BindVariableBlock(py::module_& autodiff,
                       py::class_<VariableBlock<VariableMatrix>>& cls);

void BindGradient(py::class_<Gradient>& cls);
void BindHessian(py::class_<Hessian>& cls);
void BindJacobian(py::class_<Jacobian>& cls);

void BindEqualityConstraints(py::class_<EqualityConstraints>& cls);
void BindInequalityConstraints(py::class_<InequalityConstraints>& cls);

void BindSolverExitCondition(py::enum_<SolverExitCondition>& e);
void BindSolverIterationInfo(py::class_<SolverIterationInfo>& cls);
void BindSolverStatus(py::class_<SolverStatus>& cls);

void BindOptimizationProblem(py::class_<OptimizationProblem>& cls);

}  // namespace sleipnir
