// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <concepts>
#include <initializer_list>
#include <type_traits>
#include <vector>

#include "sleipnir/autodiff/Variable.hpp"
#include "sleipnir/util/Assert.hpp"
#include "sleipnir/util/Concepts.hpp"
#include "sleipnir/util/SymbolExports.hpp"
#include "sleipnir/util/small_vector.hpp"

namespace sleipnir {

/**
 * Make a list of constraints.
 *
 * The standard form for equality constraints is c(x) = 0, and the standard form
 * for inequality constraints is c(x) ≥ 0. This function takes constraints of
 * the form lhs = rhs or lhs ≥ rhs and converts them to lhs - rhs = 0 or
 * lhs - rhs ≥ 0.
 *
 * @param lhs Left-hand side.
 * @param rhs Right-hand side.
 */
template <typename LHS, typename RHS>
  requires(ScalarLike<std::decay_t<LHS>> || MatrixLike<std::decay_t<LHS>>) &&
          (ScalarLike<std::decay_t<RHS>> || MatrixLike<std::decay_t<RHS>>) &&
          (!std::same_as<std::decay_t<LHS>, double> ||
           !std::same_as<std::decay_t<RHS>, double>)
small_vector<Variable> MakeConstraints(LHS&& lhs, RHS&& rhs) {
  small_vector<Variable> constraints;

  if constexpr (ScalarLike<std::decay_t<LHS>> &&
                ScalarLike<std::decay_t<RHS>>) {
    constraints.emplace_back(lhs - rhs);
  } else if constexpr (ScalarLike<std::decay_t<LHS>> &&
                       MatrixLike<std::decay_t<RHS>>) {
    int rows;
    int cols;
    if constexpr (EigenMatrixLike<std::decay_t<RHS>>) {
      rows = rhs.rows();
      cols = rhs.cols();
    } else {
      rows = rhs.Rows();
      cols = rhs.Cols();
    }

    constraints.reserve(rows * cols);

    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        // Make right-hand side zero
        constraints.emplace_back(lhs - rhs(row, col));
      }
    }
  } else if constexpr (MatrixLike<std::decay_t<LHS>> &&
                       ScalarLike<std::decay_t<RHS>>) {
    int rows;
    int cols;
    if constexpr (EigenMatrixLike<std::decay_t<LHS>>) {
      rows = lhs.rows();
      cols = lhs.cols();
    } else {
      rows = lhs.Rows();
      cols = lhs.Cols();
    }

    constraints.reserve(rows * cols);

    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        // Make right-hand side zero
        constraints.emplace_back(lhs(row, col) - rhs);
      }
    }
  } else if constexpr (MatrixLike<std::decay_t<LHS>> &&
                       MatrixLike<std::decay_t<RHS>>) {
    int lhsRows;
    int lhsCols;
    if constexpr (EigenMatrixLike<std::decay_t<LHS>>) {
      lhsRows = lhs.rows();
      lhsCols = lhs.cols();
    } else {
      lhsRows = lhs.Rows();
      lhsCols = lhs.Cols();
    }

    [[maybe_unused]]
    int rhsRows;
    [[maybe_unused]]
    int rhsCols;
    if constexpr (EigenMatrixLike<std::decay_t<RHS>>) {
      rhsRows = rhs.rows();
      rhsCols = rhs.cols();
    } else {
      rhsRows = rhs.Rows();
      rhsCols = rhs.Cols();
    }

    Assert(lhsRows == rhsRows && lhsCols == rhsCols);
    constraints.reserve(lhsRows * lhsCols);

    for (int row = 0; row < lhsRows; ++row) {
      for (int col = 0; col < lhsCols; ++col) {
        // Make right-hand side zero
        constraints.emplace_back(lhs(row, col) - rhs(row, col));
      }
    }
  }

  return constraints;
}

/**
 * A vector of equality constraints of the form cₑ(x) = 0.
 */
struct SLEIPNIR_DLLEXPORT EqualityConstraints {
  /// A vector of scalar equality constraints.
  small_vector<Variable> constraints;

  /**
   * Concatenates multiple equality constraints.
   *
   * @param equalityConstraints The list of EqualityConstraints to concatenate.
   */
  EqualityConstraints(
      std::initializer_list<EqualityConstraints> equalityConstraints) {
    for (const auto& elem : equalityConstraints) {
      constraints.insert(constraints.end(), elem.constraints.begin(),
                         elem.constraints.end());
    }
  }

  /**
   * Concatenates multiple equality constraints.
   *
   * This overload is for Python bindings only.
   *
   * @param equalityConstraints The list of EqualityConstraints to concatenate.
   */
  explicit EqualityConstraints(
      const std::vector<EqualityConstraints>& equalityConstraints) {
    for (const auto& elem : equalityConstraints) {
      constraints.insert(constraints.end(), elem.constraints.begin(),
                         elem.constraints.end());
    }
  }

  /**
   * Constructs an equality constraint from a left and right side.
   *
   * The standard form for equality constraints is c(x) = 0. This function takes
   * a constraint of the form lhs = rhs and converts it to lhs - rhs = 0.
   *
   * @param lhs Left-hand side.
   * @param rhs Right-hand side.
   */
  template <typename LHS, typename RHS>
    requires(ScalarLike<std::decay_t<LHS>> || MatrixLike<std::decay_t<LHS>>) &&
            (ScalarLike<std::decay_t<RHS>> || MatrixLike<std::decay_t<RHS>>) &&
            (!std::same_as<std::decay_t<LHS>, double> ||
             !std::same_as<std::decay_t<RHS>, double>)
  EqualityConstraints(LHS&& lhs, RHS&& rhs)
      : constraints{MakeConstraints(lhs, rhs)} {}

  /**
   * Implicit conversion operator to bool.
   */
  operator bool() {  // NOLINT
    return std::all_of(
        constraints.begin(), constraints.end(),
        [](auto& constraint) { return constraint.Value() == 0.0; });
  }
};

/**
 * A vector of inequality constraints of the form cᵢ(x) ≥ 0.
 */
struct SLEIPNIR_DLLEXPORT InequalityConstraints {
  /// A vector of scalar inequality constraints.
  small_vector<Variable> constraints;

  /**
   * Concatenates multiple inequality constraints.
   *
   * @param inequalityConstraints The list of InequalityConstraints to
   * concatenate.
   */
  InequalityConstraints(
      std::initializer_list<InequalityConstraints> inequalityConstraints) {
    for (const auto& elem : inequalityConstraints) {
      constraints.insert(constraints.end(), elem.constraints.begin(),
                         elem.constraints.end());
    }
  }

  /**
   * Concatenates multiple inequality constraints.
   *
   * This overload is for Python bindings only.
   *
   * @param inequalityConstraints The list of InequalityConstraints to
   * concatenate.
   */
  explicit InequalityConstraints(
      const std::vector<InequalityConstraints>& inequalityConstraints) {
    for (const auto& elem : inequalityConstraints) {
      constraints.insert(constraints.end(), elem.constraints.begin(),
                         elem.constraints.end());
    }
  }

  /**
   * Constructs an inequality constraint from a left and right side.
   *
   * The standard form for inequality constraints is c(x) ≥ 0. This function
   * takes a constraints of the form lhs ≥ rhs and converts it to lhs - rhs ≥ 0.
   *
   * @param lhs Left-hand side.
   * @param rhs Right-hand side.
   */
  template <typename LHS, typename RHS>
    requires(ScalarLike<std::decay_t<LHS>> || MatrixLike<std::decay_t<LHS>>) &&
            (ScalarLike<std::decay_t<RHS>> || MatrixLike<std::decay_t<RHS>>) &&
            (!std::same_as<std::decay_t<LHS>, double> ||
             !std::same_as<std::decay_t<RHS>, double>)
  InequalityConstraints(LHS&& lhs, RHS&& rhs)
      : constraints{MakeConstraints(lhs, rhs)} {}

  /**
   * Implicit conversion operator to bool.
   */
  operator bool() {  // NOLINT
    return std::all_of(
        constraints.begin(), constraints.end(),
        [](auto& constraint) { return constraint.Value() >= 0.0; });
  }
};

/**
 * Equality operator that returns an equality constraint for two Variables.
 *
 * @param lhs Left-hand side.
 * @param rhs Left-hand side.
 */
template <typename LHS, typename RHS>
  requires(ScalarLike<std::decay_t<LHS>> || MatrixLike<std::decay_t<LHS>>) &&
          (ScalarLike<std::decay_t<RHS>> || MatrixLike<std::decay_t<RHS>>) &&
          (!std::same_as<std::decay_t<LHS>, double> ||
           !std::same_as<std::decay_t<RHS>, double>)
EqualityConstraints operator==(LHS&& lhs, RHS&& rhs) {
  return EqualityConstraints{lhs, rhs};
}

/**
 * Less-than comparison operator that returns an inequality constraint for two
 * Variables.
 *
 * @param lhs Left-hand side.
 * @param rhs Left-hand side.
 */
template <typename LHS, typename RHS>
  requires(ScalarLike<std::decay_t<LHS>> || MatrixLike<std::decay_t<LHS>>) &&
          (ScalarLike<std::decay_t<RHS>> || MatrixLike<std::decay_t<RHS>>) &&
          (!std::same_as<std::decay_t<LHS>, double> ||
           !std::same_as<std::decay_t<RHS>, double>)
InequalityConstraints operator<(LHS&& lhs, RHS&& rhs) {
  return rhs >= lhs;
}

/**
 * Less-than-or-equal-to comparison operator that returns an inequality
 * constraint for two Variables.
 *
 * @param lhs Left-hand side.
 * @param rhs Left-hand side.
 */
template <typename LHS, typename RHS>
  requires(ScalarLike<std::decay_t<LHS>> || MatrixLike<std::decay_t<LHS>>) &&
          (ScalarLike<std::decay_t<RHS>> || MatrixLike<std::decay_t<RHS>>) &&
          (!std::same_as<std::decay_t<LHS>, double> ||
           !std::same_as<std::decay_t<RHS>, double>)
InequalityConstraints operator<=(LHS&& lhs, RHS&& rhs) {
  return rhs >= lhs;
}

/**
 * Greater-than comparison operator that returns an inequality constraint for
 * two Variables.
 *
 * @param lhs Left-hand side.
 * @param rhs Left-hand side.
 */
template <typename LHS, typename RHS>
  requires(ScalarLike<std::decay_t<LHS>> || MatrixLike<std::decay_t<LHS>>) &&
          (ScalarLike<std::decay_t<RHS>> || MatrixLike<std::decay_t<RHS>>) &&
          (!std::same_as<std::decay_t<LHS>, double> ||
           !std::same_as<std::decay_t<RHS>, double>)
InequalityConstraints operator>(LHS&& lhs, RHS&& rhs) {
  return lhs >= rhs;
}

/**
 * Greater-than-or-equal-to comparison operator that returns an inequality
 * constraint for two Variables.
 *
 * @param lhs Left-hand side.
 * @param rhs Left-hand side.
 */
template <typename LHS, typename RHS>
  requires(ScalarLike<std::decay_t<LHS>> || MatrixLike<std::decay_t<LHS>>) &&
          (ScalarLike<std::decay_t<RHS>> || MatrixLike<std::decay_t<RHS>>) &&
          (!std::same_as<std::decay_t<LHS>, double> ||
           !std::same_as<std::decay_t<RHS>, double>)
InequalityConstraints operator>=(LHS&& lhs, RHS&& rhs) {
  return InequalityConstraints{lhs, rhs};
}

}  // namespace sleipnir
