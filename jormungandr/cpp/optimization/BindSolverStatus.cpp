// Copyright (c) Sleipnir contributors

#include <pybind11/pybind11.h>
#include <sleipnir/optimization/SolverStatus.hpp>

#include "Docstrings.hpp"

namespace py = pybind11;

namespace sleipnir {

void BindSolverStatus(py::class_<SolverStatus>& cls) {
  using namespace py::literals;

  cls.def(py::init<>());
  cls.def(py::init<ExpressionType, ExpressionType, ExpressionType,
                   SolverExitCondition, double>(),
          py::arg_v("cost_function_type", ExpressionType::kNone,
                    "ExpressionType.NONE"),
          py::arg_v("equality_constraint_type", ExpressionType::kNone,
                    "ExpressionType.NONE"),
          py::arg_v("inequality_constraint_type", ExpressionType::kNone,
                    "ExpressionType.NONE"),
          py::arg_v("exit_condition", SolverExitCondition::kSuccess,
                    "SolverExitCondition.SUCCESS"),
          "cost"_a = 0.0);
  cls.def_readwrite("cost_function_type", &SolverStatus::costFunctionType,
                    DOC(sleipnir, SolverStatus, costFunctionType));
  cls.def_readwrite("equality_constraint_type",
                    &SolverStatus::equalityConstraintType,
                    DOC(sleipnir, SolverStatus, equalityConstraintType));
  cls.def_readwrite("inequality_constraint_type",
                    &SolverStatus::inequalityConstraintType,
                    DOC(sleipnir, SolverStatus, inequalityConstraintType));
  cls.def_readwrite("exit_condition", &SolverStatus::exitCondition,
                    DOC(sleipnir, SolverStatus, exitCondition));
  cls.def_readwrite("cost", &SolverStatus::cost,
                    DOC(sleipnir, SolverStatus, cost));
}

}  // namespace sleipnir
