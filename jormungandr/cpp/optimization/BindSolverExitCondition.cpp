// Copyright (c) Sleipnir contributors

#include "optimization/BindSolverExitCondition.hpp"

#include <sleipnir/optimization/SolverExitCondition.hpp>

namespace py = pybind11;

namespace sleipnir {

void BindSolverExitCondition(py::module_& optimization) {
  py::enum_<SolverExitCondition> e{optimization, "SolverExitCondition"};
  e.value("SUCCESS", SolverExitCondition::kSuccess);
  e.value("SOLVED_TO_ACCEPTABLE_TOLERANCE",
          SolverExitCondition::kSolvedToAcceptableTolerance);
  e.value("CALLBACK_REQUESTED_STOP",
          SolverExitCondition::kCallbackRequestedStop);
  e.value("TOO_FEW_DOFS", SolverExitCondition::kTooFewDOFs);
  e.value("LOCALLY_INFEASIBLE", SolverExitCondition::kLocallyInfeasible);
  e.value("BAD_SEARCH_DIRECTION", SolverExitCondition::kBadSearchDirection);
  e.value("MAX_SEARCH_DIRECTION_TOO_SMALL",
          SolverExitCondition::kMaxSearchDirectionTooSmall);
  e.value("DIVERGING_ITERATES", SolverExitCondition::kDivergingIterates);
  e.value("MAX_ITERATIONS_EXCEEDED",
          SolverExitCondition::kMaxIterationsExceeded);
  e.value("MAX_WALL_CLOCK_TIME_EXCEEDED",
          SolverExitCondition::kMaxWallClockTimeExceeded);
}

}  // namespace sleipnir
