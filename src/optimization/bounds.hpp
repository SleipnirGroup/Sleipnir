// Copyright (c) Sleipnir contributors

#pragma once

#include "sleipnir/util/small_vector.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/util/assert.hpp"

#include <span>
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/SparseCore>

// See docs/algorithms.md#Works_cited for citation definitions

namespace slp {

//
/**
 * A "bound constraint" is any linear constraint in one scalar variable.
 * Computes which constraints, if any, are bound constraints, whether or not
 * they're feasible (given previously encountered bounds), and the tightest
 * bounds on each decision variable.
 *
 * @param decisionVariables Decision variables corresponding to each column of
 *   A_i.
 * @param inequalityConstraints Variables representing the left-hand side of
 *   cᵢ(decisionVariables) ≤ 0.
 * @param A_i The Jacobian of inequalityConstraints wrt decisionVariables,
 *   stored row-major*; in practice, since we typically store Jacobians
 *   column-major, the user of this function must perform a transpose.
 */
inline std::tuple<small_vector<Eigen::Index>,
                  small_vector<std::pair<double, double>>,
                  small_vector<std::pair<Eigen::Index, Eigen::Index>>>
get_bounds(const std::span<Variable> decision_variables,
           const std::span<Variable> inequality_constraints,
           const Eigen::SparseMatrix<double, Eigen::RowMajor>& A_i) {
  // A blocked, out-of-place transpose should be much faster than traversing row
  // major on a column major matrix unless we have few linear constraints (using
  // a heuristic to choose between this and staying column major based on the
  // number of constraints would be an easy performance improvement.)
  // Eigen::SparseMatrix<double, Eigen::RowMajor> row_major_A_i{A_i};
  // NB: Casting to long is unspecified if the size of decisionVariable.size()
  // is greater than the max long value, but then we wouldn't be able to fill
  // A_i correctly anyway.
  assert(static_cast<long>(decision_variables.size()) == A_i.innerSize());
  assert(static_cast<long>(inequality_constraints.size()) == A_i.outerSize());

  // Maps each decision variable's index to the indices of its upper and lower
  // bounds if they exist, or NO_BOUND if they do not; used only for bookkeeping
  // in order to compute conflicting bounds
  static constexpr Eigen::Index NO_BOUND = -1;
  small_vector<std::pair<Eigen::Index, Eigen::Index>>
      decision_var_indices_to_constraint_indices{decision_variables.size(),
                                                 {NO_BOUND, NO_BOUND}};
  // Lists pairs of indices of bound constraints in the inequality constraint
  // list that conflict with each other
  small_vector<std::pair<Eigen::Index, Eigen::Index>> conflicting_bound_indices;

  // Maps each decision variable's index to its upper and lower bounds
  small_vector<std::pair<double, double>> decision_var_indices_to_bounds{
      decision_variables.size(),
      {-std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::infinity()}};

  // Lists the indices of bound constraints in the inequality
  // constraint list, including redundant bound constraints
  small_vector<Eigen::Index> bound_constraint_indices;

  for (decltype(inequality_constraints)::size_type constraint_index = 0;
       constraint_index < inequality_constraints.size(); constraint_index++) {
    // A constraint is a bound iff it is linear and its gradient has a
    // single nonzero value.
    if (inequality_constraints[constraint_index].type() !=
        ExpressionType::LINEAR) {
      continue;
    }
    const Eigen::SparseVector<double>& row_A_i =
        A_i.innerVector(constraint_index);
    const auto non_zeros = row_A_i.nonZeros();
    assert(non_zeros != 0);
    if (non_zeros > 1) {
      // Constraint is in more than one variable.
      continue;
    }

    // Claim: The bound is given by a bound constraint is the constraint
    // evaluated at zero divided by the nonzero element of the constraint's
    // gradient.
    // Proof: If c(x) is a bound constraint, then by definition c is a linear
    // function in one variable, hence there exist a, b ∈ ℝ s.t. c(x) = axᵢ + b
    // and a ≠ 0. The gradient of c is then aeᵢ (where eᵢ denotes the i-th basis
    // element), and c(0) = b. If c(x) ≤ 0, then since either a < 0 or a > 0, we
    // have either x ≥ -b/a or x ≤ -b/a, respectively. ∎
    Eigen::SparseVector<double>::InnerIterator row_iter(row_A_i);
    const auto constraint_coefficient =
        row_iter
            .value();  // The nonzero value of the j-th constraint's gradient.
    const auto decision_variable_index = row_iter.index();
    const auto decision_variable_value =
        decision_variables[decision_variable_index].value();
    double constraint_constant;
    // We need to evaluate this constraint at zero
    if (decision_variable_value != 0) {
      decision_variables[decision_variable_index].set_value(0);
      constraint_constant = inequality_constraints[constraint_index].value();
      decision_variables[decision_variable_index].set_value(
          decision_variable_value);
    } else {
      constraint_constant = inequality_constraints[constraint_index].value();
    }
    assert(constraint_coefficient !=
           0);  // Shouldn't happen since the constraint is
                // supposed to be linear and not a constant.

    // Update bounds
    auto& [lower_bound, upper_bound] =
        decision_var_indices_to_bounds[decision_variable_index];
    auto& [lower_index, upper_index] =
        decision_var_indices_to_constraint_indices[decision_variable_index];
    // Assumes c(x) ≤ 0.
    const auto detected_bound = -constraint_constant / constraint_coefficient;
    if (constraint_coefficient < 0 && detected_bound > lower_bound) {
      lower_bound = detected_bound;
      lower_index = constraint_index;
    } else if (constraint_coefficient > 0 && detected_bound < upper_bound) {
      upper_bound = detected_bound;
      upper_index = constraint_index;
    }

    // Update conflicting bounds
    if (lower_bound > upper_bound) {
      conflicting_bound_indices.emplace_back(lower_index, upper_index);
    }

    // Not used in any current solver, but can be used in any algorithm that
    // explicitly controls the rate of decrease of the duals like [5], where we
    // would set wⱼ = 0 each j bound_constraint_indices
    bound_constraint_indices.emplace_back(constraint_index);
  }
  return {bound_constraint_indices, decision_var_indices_to_bounds,
          conflicting_bound_indices};
}

/**
 * Projects the decision variables onto the given bounds, while ensuring some
 * configurable distance from the boundary if possible. This is designed to
 * match the algorithm given in section 3.6 of [2].
 *
 * @param x A vector of decision variables.
 * @param bounds An array of bounds (stored [lower, upper]) for each decision
 *   variable in x (implicitly a map from decision variable index to bound).
 * @param κ_1 A constant controlling distance from the lower or upper bound when
 *   the difference between the upper and lower bound is small.
 * @param κ_2 A constant controlling distance from the lower or upper bound when
 *   the difference between the upper and lower bound is large (including when
 *   one of the bounds is ±∞).
 */
template <typename Derived>
  requires(static_cast<bool>(Eigen::DenseBase<Derived>::IsVectorAtCompileTime))
inline void project_onto_bounds(
    Eigen::DenseBase<Derived>& x,
    const std::span<std::pair<typename Eigen::DenseBase<Derived>::Scalar,
                              typename Eigen::DenseBase<Derived>::Scalar>>
        bounds,
    const typename Eigen::DenseBase<Derived>::Scalar κ_1 = 1e-2,
    const typename Eigen::DenseBase<Derived>::Scalar κ_2 = 1e-2) {
  assert(κ_1 > 0 && κ_2 > 0 && κ_2 < 0.5);

  Eigen::Index idx = 0;
  for (const auto& [lower, upper] : bounds) {
    typename Eigen::DenseBase<Derived>::Scalar& x_i = x[idx++];

    // We assume that bound infeasibility is handled elsewhere.
    assert(lower <= upper);

    // See B.2 in [5] and section 3.6 in [2]
    if (std::isfinite(lower) && std::isfinite(upper)) {
      auto p_L =
          std::min(κ_1 * std::max(1.0, std::abs(lower)), κ_2 * (upper - lower));
      auto p_U =
          std::min(κ_1 * std::max(1.0, std::abs(upper)), κ_2 * (upper - lower));
      x_i = std::min(std::max(lower + p_L, x_i), upper - p_U);
    } else if (std::isfinite(lower)) {
      x_i = std::max(x_i, lower + κ_1 * std::max(1.0, std::abs(lower)));
    } else if (std::isfinite(upper)) {
      x_i = std::min(x_i, upper - κ_1 * std::max(1.0, std::abs(upper)));
    }
  }
}
}  // namespace slp
