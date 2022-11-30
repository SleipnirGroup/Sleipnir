// Copyright (c) Joshua Nichols and Tyler Veness

#pragma once

#include <vector>

#include "sleipnir/SymbolExports.hpp"
#include "sleipnir/autodiff/Variable.hpp"
#include "sleipnir/autodiff/VariableMatrix.hpp"

namespace sleipnir {

/**
 * A vector of equality constraints of the form cₑ(x) = 0.
 */
struct SLEIPNIR_DLLEXPORT EqualityConstraints {
  /// A vector of scalar equality constraints.
  std::vector<Variable> constraints;
};

/**
 * A vector of inequality constraints of the form cᵢ(x) ≥ 0.
 */
struct SLEIPNIR_DLLEXPORT InequalityConstraints {
  /// A vector of scalar inequality constraints.
  std::vector<Variable> constraints;
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
