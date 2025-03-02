// Copyright (c) Sleipnir contributors

#include <nanobind/nanobind.h>
#include <sleipnir/optimization/solver_exit_condition.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_solver_exit_condition(nb::enum_<SolverExitCondition>& e) {
  e.value("SUCCESS", SolverExitCondition::SUCCESS,
          DOC(slp, SolverExitCondition, SUCCESS));
  e.value("SOLVED_TO_ACCEPTABLE_TOLERANCE",
          SolverExitCondition::SOLVED_TO_ACCEPTABLE_TOLERANCE,
          DOC(slp, SolverExitCondition, SOLVED_TO_ACCEPTABLE_TOLERANCE));
  e.value("CALLBACK_REQUESTED_STOP",
          SolverExitCondition::CALLBACK_REQUESTED_STOP,
          DOC(slp, SolverExitCondition, CALLBACK_REQUESTED_STOP));
  e.value("TOO_FEW_DOFS", SolverExitCondition::TOO_FEW_DOFS,
          DOC(slp, SolverExitCondition, TOO_FEW_DOFS));
  e.value("LOCALLY_INFEASIBLE", SolverExitCondition::LOCALLY_INFEASIBLE,
          DOC(slp, SolverExitCondition, LOCALLY_INFEASIBLE));
  e.value("FACTORIZATION_FAILED", SolverExitCondition::FACTORIZATION_FAILED,
          DOC(slp, SolverExitCondition, FACTORIZATION_FAILED));
  e.value("LINE_SEARCH_FAILED", SolverExitCondition::LINE_SEARCH_FAILED,
          DOC(slp, SolverExitCondition, LINE_SEARCH_FAILED));
  e.value("NONFINITE_INITIAL_COST_OR_CONSTRAINTS",
          SolverExitCondition::NONFINITE_INITIAL_COST_OR_CONSTRAINTS,
          DOC(slp, SolverExitCondition, NONFINITE_INITIAL_COST_OR_CONSTRAINTS));
  e.value("DIVERGING_ITERATES", SolverExitCondition::DIVERGING_ITERATES,
          DOC(slp, SolverExitCondition, DIVERGING_ITERATES));
  e.value("MAX_ITERATIONS_EXCEEDED",
          SolverExitCondition::MAX_ITERATIONS_EXCEEDED,
          DOC(slp, SolverExitCondition, MAX_ITERATIONS_EXCEEDED));
  e.value("TIMEOUT", SolverExitCondition::TIMEOUT,
          DOC(slp, SolverExitCondition, TIMEOUT));
}

}  // namespace slp
