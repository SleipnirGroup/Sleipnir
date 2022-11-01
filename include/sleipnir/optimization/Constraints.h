// Copyright (c) Joshua Nichols and Tyler Veness

#pragma once

#include <vector>

#include "sleipnir/SymbolExports.h"
#include "sleipnir/autodiff/Variable.h"
#include "sleipnir/optimization/VariableMatrix.h"

namespace sleipnir {

/**
 * An equality constraint has the form c(x) = 0.
 */
struct SLEIPNIR_DLLEXPORT EqualityConstraints {
  std::vector<autodiff::Variable> constraints;
};

/**
 * An inequality constraint has the form c(x) ≥ 0.
 */
struct SLEIPNIR_DLLEXPORT InequalityConstraints {
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
