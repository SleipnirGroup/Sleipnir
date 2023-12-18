// Copyright (c) Sleipnir contributors

#pragma once

#include <vector>

#include "sleipnir/autodiff/Variable.hpp"
#include "sleipnir/autodiff/VariableMatrix.hpp"
#include "sleipnir/util/SymbolExports.hpp"

namespace sleipnir {

/**
 * A vector of equality constraints of the form cₑ(x) = 0.
 */
struct SLEIPNIR_DLLEXPORT EqualityConstraints {
  /// A vector of scalar equality constraints.
  std::vector<Variable> constraints;

  /**
   * Constructs an equality constraint from a left and right side.
   *
   * The standard form for equality constraints is c(x) = 0. This function takes
   * a constraint of the form lhs = rhs and converts it to lhs - rhs = 0.
   *
   * @param lhs Left-hand side.
   * @param rhs Right-hand side.
   */
  EqualityConstraints(const Variable& lhs, const Variable& rhs);

  /**
   * Constructs an equality constraint from a left and right side.
   *
   * The standard form for equality constraints is c(x) = 0. This function takes
   * a constraint of the form lhs = rhs and converts it to lhs - rhs = 0.
   *
   * @param lhs Left-hand side.
   * @param rhs Right-hand side.
   */
  EqualityConstraints(const VariableMatrix& lhs, const VariableMatrix& rhs);

  /**
   * Implicit conversion operator to bool.
   */
  operator bool() const;  // NOLINT
};

/**
 * A vector of inequality constraints of the form cᵢ(x) ≥ 0.
 */
struct SLEIPNIR_DLLEXPORT InequalityConstraints {
  /// A vector of scalar inequality constraints.
  std::vector<Variable> constraints;

  /**
   * Constructs an inequality constraint from a left and right side.
   *
   * The standard form for inequality constraints is c(x) ≥ 0. This function
   * takes a constraints of the form lhs ≥ rhs and converts it to lhs - rhs ≥ 0.
   *
   * @param lhs Left-hand side.
   * @param rhs Right-hand side.
   */
  InequalityConstraints(const Variable& lhs, const Variable& rhs);

  /**
   * Constructs an inequality constraint from a left and right side.
   *
   * The standard form for inequality constraints is c(x) ≥ 0. This function
   * takes a constraints of the form lhs ≥ rhs and converts it to lhs - rhs ≥ 0.
   *
   * @param lhs Left-hand side.
   * @param rhs Right-hand side.
   */
  InequalityConstraints(const VariableMatrix& lhs, const VariableMatrix& rhs);

  /**
   * Implicit conversion operator to bool.
   */
  operator bool() const;  // NOLINT
};

/**
 * Equality operator that returns an equality constraint for two Variables.
 *
 * @param lhs Left-hand side.
 * @param rhs Left-hand side.
 */
SLEIPNIR_DLLEXPORT EqualityConstraints operator==(const Variable& lhs,
                                                  const Variable& rhs);

/**
 * Less-than comparison operator that returns an inequality constraint for two
 * Variables.
 *
 * @param lhs Left-hand side.
 * @param rhs Left-hand side.
 */
SLEIPNIR_DLLEXPORT InequalityConstraints operator<(const Variable& lhs,
                                                   const Variable& rhs);

/**
 * Less-than-or-equal-to comparison operator that returns an inequality
 * constraint for two Variables.
 *
 * @param lhs Left-hand side.
 * @param rhs Left-hand side.
 */
SLEIPNIR_DLLEXPORT InequalityConstraints operator<=(const Variable& lhs,
                                                    const Variable& rhs);

/**
 * Greater-than comparison operator that returns an inequality constraint for
 * two Variables.
 *
 * @param lhs Left-hand side.
 * @param rhs Left-hand side.
 */
SLEIPNIR_DLLEXPORT InequalityConstraints operator>(const Variable& lhs,
                                                   const Variable& rhs);

/**
 * Greater-than-or-equal-to comparison operator that returns an inequality
 * constraint for two Variables.
 *
 * @param lhs Left-hand side.
 * @param rhs Left-hand side.
 */
SLEIPNIR_DLLEXPORT InequalityConstraints operator>=(const Variable& lhs,
                                                    const Variable& rhs);

/**
 * Equality operator that returns an equality constraint for two
 * VariableMatrices.
 *
 * @param lhs Left-hand side.
 * @param rhs Left-hand side.
 */
SLEIPNIR_DLLEXPORT EqualityConstraints operator==(const VariableMatrix& lhs,
                                                  const VariableMatrix& rhs);

/**
 * Less-than comparison operator that returns an inequality constraint for two
 * VariableMatrices.
 *
 * @param lhs Left-hand side.
 * @param rhs Left-hand side.
 */
SLEIPNIR_DLLEXPORT InequalityConstraints operator<(const VariableMatrix& lhs,
                                                   const VariableMatrix& rhs);

/**
 * Less-than-or-equal-to comparison operator that returns an inequality
 * constraint for two VariableMatrices.
 *
 * @param lhs Left-hand side.
 * @param rhs Left-hand side.
 */
SLEIPNIR_DLLEXPORT InequalityConstraints operator<=(const VariableMatrix& lhs,
                                                    const VariableMatrix& rhs);

/**
 * Greater-than comparison operator that returns an inequality constraint for
 * two VariableMatrices.
 *
 * @param lhs Left-hand side.
 * @param rhs Left-hand side.
 */
SLEIPNIR_DLLEXPORT InequalityConstraints operator>(const VariableMatrix& lhs,
                                                   const VariableMatrix& rhs);

/**
 * Greater-than-or-equal-to comparison operator that returns an inequality
 * constraint for two VariableMatrices.
 *
 * @param lhs Left-hand side.
 * @param rhs Left-hand side.
 */
SLEIPNIR_DLLEXPORT InequalityConstraints operator>=(const VariableMatrix& lhs,
                                                    const VariableMatrix& rhs);

}  // namespace sleipnir
