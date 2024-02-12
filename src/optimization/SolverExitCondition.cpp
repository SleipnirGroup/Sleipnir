// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/SolverExitCondition.hpp"

#include <ostream>

namespace sleipnir {

std::ostream& operator<<(std::ostream& os,
                         const SolverExitCondition& exitCondition) {
  using enum sleipnir::SolverExitCondition;

  switch (exitCondition) {
    case kSuccess:
      os << "kSuccess";
      break;
    case kSolvedToAcceptableTolerance:
      os << "kSolvedToAcceptableTolerance";
      break;
    case kCallbackRequestedStop:
      os << "kCallbackRequestedStop";
      break;
    case kTooFewDOFs:
      os << "kTooFewDOFs";
      break;
    case kLocallyInfeasible:
      os << "kLocallyInfeasible";
      break;
    case kFeasibilityRestorationFailed:
      os << "kFeasibilityRestorationFailed";
      break;
    case kNonfiniteInitialCostOrConstraints:
      os << "kNonfiniteInitialCostOrConstraints";
      break;
    case kDivergingIterates:
      os << "kDivergingIterates";
      break;
    case kMaxIterationsExceeded:
      os << "kMaxIterationsExceeded";
      break;
    case kMaxWallClockTimeExceeded:
      os << "kMaxWallClockTimeExceeded";
      break;
  }

  return os;
}

}  // namespace sleipnir
