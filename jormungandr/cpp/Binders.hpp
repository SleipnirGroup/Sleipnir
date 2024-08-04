// Copyright (c) Sleipnir contributors

#pragma once

#include <nanobind/nanobind.h>
#include <sleipnir/autodiff/ExpressionType.hpp>
#include <sleipnir/autodiff/Gradient.hpp>
#include <sleipnir/autodiff/Hessian.hpp>
#include <sleipnir/autodiff/Jacobian.hpp>
#include <sleipnir/autodiff/Variable.hpp>
#include <sleipnir/autodiff/VariableBlock.hpp>
#include <sleipnir/autodiff/VariableMatrix.hpp>
#include <sleipnir/control/OCPSolver.hpp>
#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <sleipnir/optimization/SolverExitCondition.hpp>
#include <sleipnir/optimization/SolverIterationInfo.hpp>
#include <sleipnir/optimization/SolverStatus.hpp>

namespace nb = nanobind;

namespace sleipnir {

void BindExpressionType(nb::enum_<ExpressionType>& e);

void BindVariable(nb::module_& autodiff, nb::class_<Variable>& cls);
void BindVariableMatrix(nb::module_& autodiff, nb::class_<VariableMatrix>& cls);
void BindVariableBlock(nb::class_<VariableBlock<VariableMatrix>>& cls);

void BindGradient(nb::class_<Gradient>& cls);
void BindHessian(nb::class_<Hessian>& cls);
void BindJacobian(nb::class_<Jacobian>& cls);

void BindEqualityConstraints(nb::class_<EqualityConstraints>& cls);
void BindInequalityConstraints(nb::class_<InequalityConstraints>& cls);

void BindSolverExitCondition(nb::enum_<SolverExitCondition>& e);
void BindSolverIterationInfo(nb::class_<SolverIterationInfo>& cls);
void BindSolverStatus(nb::class_<SolverStatus>& cls);

void BindOptimizationProblem(nb::class_<OptimizationProblem>& cls);

void BindOCPSolver(nb::enum_<TranscriptionMethod>& transcription_method,
                   nb::enum_<DynamicsType>& dynamics_type,
                   nb::enum_<TimestepMethod>& timestep_method,
                   nb::class_<OCPSolver, OptimizationProblem>& cls);

}  // namespace sleipnir
