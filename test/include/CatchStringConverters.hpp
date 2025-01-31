// Copyright (c) Sleipnir contributors

#pragma once

#include <string>

#include <catch2/catch_tostring.hpp>
#include <sleipnir/autodiff/ExpressionType.hpp>
#include <sleipnir/optimization/SolverExitCondition.hpp>

namespace Catch {

template <>
struct StringMaker<sleipnir::ExpressionType> {
  static std::string convert(const sleipnir::ExpressionType& type) {
    using enum sleipnir::ExpressionType;

    switch (type) {
      case kNone:
        return "kNone";
      case kConstant:
        return "kConstant";
      case kLinear:
        return "kLinear";
      case kQuadratic:
        return "kQuadratic";
      case kNonlinear:
        return "kNonlinear";
    }

    return "";
  }
};

template <>
struct StringMaker<sleipnir::SolverExitCondition> {
  static std::string convert(
      const sleipnir::SolverExitCondition& exitCondition) {
    using enum sleipnir::SolverExitCondition;

    switch (exitCondition) {
      case kSuccess:
        return "kSuccess";
      case kSolvedToAcceptableTolerance:
        return "kSolvedToAcceptableTolerance";
      case kCallbackRequestedStop:
        return "kCallbackRequestedStop";
      case kTooFewDOFs:
        return "kTooFewDOFs";
      case kLocallyInfeasible:
        return "kLocallyInfeasible";
      case kFactorizationFailed:
        return "kFactorizationFailed";
      case kLineSearchFailed:
        return "kLineSearchFailed";
      case kNonfiniteInitialCostOrConstraints:
        return "kNonfiniteInitialCostOrConstraints";
      case kDivergingIterates:
        return "kDivergingIterates";
      case kMaxIterationsExceeded:
        return "kMaxIterationsExceeded";
      case kTimeout:
        return "kTimeout";
    }

    return "";
  }
};

}  // namespace Catch
