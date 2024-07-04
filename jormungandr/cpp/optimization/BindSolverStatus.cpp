// Copyright (c) Sleipnir contributors

#include <nanobind/nanobind.h>
#include <sleipnir/optimization/SolverStatus.hpp>

#include "Docstrings.hpp"

namespace nb = nanobind;

namespace sleipnir {

void BindSolverStatus(nb::class_<SolverStatus>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<>());
  cls.def(nb::init<ExpressionType, ExpressionType, ExpressionType,
                   SolverExitCondition, double>(),
          "cost_function_type"_a = ExpressionType::kNone,
          "equality_constraint_type"_a = ExpressionType::kNone,
          "inequality_constraint_type"_a = ExpressionType::kNone,
          "exit_condition"_a = SolverExitCondition::kSuccess, "cost"_a = 0.0);
  cls.def_rw("cost_function_type", &SolverStatus::costFunctionType,
             DOC(sleipnir, SolverStatus, costFunctionType));
  cls.def_rw("equality_constraint_type", &SolverStatus::equalityConstraintType,
             DOC(sleipnir, SolverStatus, equalityConstraintType));
  cls.def_rw("inequality_constraint_type",
             &SolverStatus::inequalityConstraintType,
             DOC(sleipnir, SolverStatus, inequalityConstraintType));
  cls.def_rw("exit_condition", &SolverStatus::exitCondition,
             DOC(sleipnir, SolverStatus, exitCondition));
  cls.def_rw("cost", &SolverStatus::cost, DOC(sleipnir, SolverStatus, cost));
}

}  // namespace sleipnir
