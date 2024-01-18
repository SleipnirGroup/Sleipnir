// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/SolverExitCondition.hpp"

#include <ostream>

namespace sleipnir {

void PrintTo(const SolverExitCondition& exitCondition, std::ostream* os) {
  using enum sleipnir::SolverExitCondition;

  switch (exitCondition) {
    case kSuccess:
      *os << "kSuccess";
      break;
    case kSolvedToAcceptableTolerance:
      *os << "kSolvedToAcceptableTolerance";
      break;
    case kCallbackRequestedStop:
      *os << "kCallbackRequestedStop";
      break;
    case kTooFewDOFs:
      *os << "kTooFewDOFs";
      break;
    case kLocallyInfeasible:
      *os << "kLocallyInfeasible";
      break;
    case kFeasibilityRestorationFailed:
      *os << "kFeasibilityRestorationFailed";
      break;
    case kMaxSearchDirectionTooSmall:
      *os << "kMaxSearchDirectionTooSmall";
      break;
    case kNonfiniteInitialCostOrConstraints:
      *os << "kNonfiniteInitialCostOrConstraints";
      break;
    case kDivergingIterates:
      *os << "kDivergingIterates";
      break;
    case kMaxIterationsExceeded:
      *os << "kMaxIterationsExceeded";
      break;
    case kMaxWallClockTimeExceeded:
      *os << "kMaxWallClockTimeExceeded";
      break;
  }
}

}  // namespace sleipnir
