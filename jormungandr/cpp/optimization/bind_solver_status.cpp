// Copyright (c) Sleipnir contributors

#include <nanobind/nanobind.h>
#include <sleipnir/optimization/solver_status.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_solver_status(nb::class_<SolverStatus>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<>());
  cls.def(nb::init<ExpressionType, ExpressionType, ExpressionType,
                   SolverExitCondition, double>(),
          "cost_function_type"_a = ExpressionType::NONE,
          "equality_constraint_type"_a = ExpressionType::NONE,
          "inequality_constraint_type"_a = ExpressionType::NONE,
          "exit_condition"_a = SolverExitCondition::SUCCESS, "cost"_a = 0.0);
  cls.def_rw("cost_function_type", &SolverStatus::cost_function_type,
             DOC(slp, SolverStatus, cost_function_type));
  cls.def_rw("equality_constraint_type",
             &SolverStatus::equality_constraint_type,
             DOC(slp, SolverStatus, equality_constraint_type));
  cls.def_rw("inequality_constraint_type",
             &SolverStatus::inequality_constraint_type,
             DOC(slp, SolverStatus, inequality_constraint_type));
  cls.def_rw("exit_condition", &SolverStatus::exit_condition,
             DOC(slp, SolverStatus, exit_condition));
  cls.def_rw("cost", &SolverStatus::cost, DOC(slp, SolverStatus, cost));
}

}  // namespace slp
