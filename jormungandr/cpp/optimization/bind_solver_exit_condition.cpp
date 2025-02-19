// Copyright (c) Sleipnir contributors

#include <nanobind/nanobind.h>
#include <sleipnir/optimization/solver_exit_condition.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace sleipnir {

void bind_solver_exit_condition(nb::enum_<SolverExitCondition>& e) {
  e.value("SUCCESS", SolverExitCondition::SUCCESS,
          DOC(sleipnir, SolverExitCondition, SUCCESS));
  e.value("SOLVED_TO_ACCEPTABLE_TOLERANCE",
          SolverExitCondition::SOLVED_TO_ACCEPTABLE_TOLERANCE,
          DOC(sleipnir, SolverExitCondition, SOLVED_TO_ACCEPTABLE_TOLERANCE));
  e.value("CALLBACK_REQUESTED_STOP",
          SolverExitCondition::CALLBACK_REQUESTED_STOP,
          DOC(sleipnir, SolverExitCondition, CALLBACK_REQUESTED_STOP));
  e.value("TOO_FEW_DOFS", SolverExitCondition::TOO_FEW_DOFS,
          DOC(sleipnir, SolverExitCondition, TOO_FEW_DOFS));
  e.value("LOCALLY_INFEASIBLE", SolverExitCondition::LOCALLY_INFEASIBLE,
          DOC(sleipnir, SolverExitCondition, LOCALLY_INFEASIBLE));
  e.value("FACTORIZATION_FAILED", SolverExitCondition::FACTORIZATION_FAILED,
          DOC(sleipnir, SolverExitCondition, FACTORIZATION_FAILED));
  e.value("LINE_SEARCH_FAILED", SolverExitCondition::LINE_SEARCH_FAILED,
          DOC(sleipnir, SolverExitCondition, LINE_SEARCH_FAILED));
  e.value("NONFINITE_INITIAL_COST_OR_CONSTRAINTS",
          SolverExitCondition::NONFINITE_INITIAL_COST_OR_CONSTRAINTS,
          DOC(sleipnir, SolverExitCondition,
              NONFINITE_INITIAL_COST_OR_CONSTRAINTS));
  e.value("DIVERGING_ITERATES", SolverExitCondition::DIVERGING_ITERATES,
          DOC(sleipnir, SolverExitCondition, DIVERGING_ITERATES));
  e.value("MAX_ITERATIONS_EXCEEDED",
          SolverExitCondition::MAX_ITERATIONS_EXCEEDED,
          DOC(sleipnir, SolverExitCondition, MAX_ITERATIONS_EXCEEDED));
  e.value("TIMEOUT", SolverExitCondition::TIMEOUT,
          DOC(sleipnir, SolverExitCondition, TIMEOUT));
}

}  // namespace sleipnir
