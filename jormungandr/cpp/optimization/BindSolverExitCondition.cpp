// Copyright (c) Sleipnir contributors

#include <pybind11/pybind11.h>
#include <sleipnir/optimization/SolverExitCondition.hpp>

#include "Docstrings.hpp"

namespace py = pybind11;

namespace sleipnir {

void BindSolverExitCondition(py::enum_<SolverExitCondition>& e) {
  e.value("SUCCESS", SolverExitCondition::kSuccess,
          DOC(sleipnir, SolverExitCondition, kSuccess));
  e.value("SOLVED_TO_ACCEPTABLE_TOLERANCE",
          SolverExitCondition::kSolvedToAcceptableTolerance,
          DOC(sleipnir, SolverExitCondition, kSolvedToAcceptableTolerance));
  e.value("CALLBACK_REQUESTED_STOP",
          SolverExitCondition::kCallbackRequestedStop,
          DOC(sleipnir, SolverExitCondition, kCallbackRequestedStop));
  e.value("TOO_FEW_DOFS", SolverExitCondition::kTooFewDOFs,
          DOC(sleipnir, SolverExitCondition, kTooFewDOFs));
  e.value("LOCALLY_INFEASIBLE", SolverExitCondition::kLocallyInfeasible,
          DOC(sleipnir, SolverExitCondition, kLocallyInfeasible));
  e.value("FEASIBILITY_RESTORATION_FAILED",
          SolverExitCondition::kFeasibilityRestorationFailed,
          DOC(sleipnir, SolverExitCondition, kFeasibilityRestorationFailed));
  e.value(
      "NONFINITE_INITIAL_COST_OR_CONSTRAINTS",
      SolverExitCondition::kNonfiniteInitialCostOrConstraints,
      DOC(sleipnir, SolverExitCondition, kNonfiniteInitialCostOrConstraints));
  e.value("DIVERGING_ITERATES", SolverExitCondition::kDivergingIterates,
          DOC(sleipnir, SolverExitCondition, kDivergingIterates));
  e.value("MAX_ITERATIONS_EXCEEDED",
          SolverExitCondition::kMaxIterationsExceeded,
          DOC(sleipnir, SolverExitCondition, kMaxIterationsExceeded));
  e.value("TIMEOUT", SolverExitCondition::kTimeout,
          DOC(sleipnir, SolverExitCondition, kTimeout));
}

}  // namespace sleipnir
