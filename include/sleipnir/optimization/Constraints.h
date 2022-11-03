// Copyright (c) Joshua Nichols and Tyler Veness

#pragma once

#include <vector>

#include "sleipnir/SymbolExports.h"
#include "sleipnir/autodiff/Variable.h"
#include "sleipnir/optimization/VariableMatrix.h"

namespace sleipnir {

/**
 * A vector of equality constraints of the form cₑ(x) = 0.
 */
struct SLEIPNIR_DLLEXPORT EqualityConstraints {
  /// A vector of scalar equality constraints.
  std::vector<autodiff::Variable> constraints;
};

/**
 * A vector of inequality constraints of the form cᵢ(x) ≥ 0.
 */
struct SLEIPNIR_DLLEXPORT InequalityConstraints {
  /// A vector of scalar inequality constraints.
  std::vector<autodiff::Variable> constraints;
};

SLEIPNIR_DLLEXPORT EqualityConstraints operator==(const VariableMatrix& lhs,
                                                  const VariableMatrix& rhs);

SLEIPNIR_DLLEXPORT InequalityConstraints operator<(const VariableMatrix& lhs,
                                                   const VariableMatrix& rhs);

SLEIPNIR_DLLEXPORT InequalityConstraints operator<=(const VariableMatrix& lhs,
                                                    const VariableMatrix& rhs);

SLEIPNIR_DLLEXPORT InequalityConstraints operator>(const VariableMatrix& lhs,
                                                   const VariableMatrix& rhs);

SLEIPNIR_DLLEXPORT InequalityConstraints operator>=(const VariableMatrix& lhs,
                                                    const VariableMatrix& rhs);

}  // namespace sleipnir
