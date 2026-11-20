// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>

// See docs/algorithms.md#Works_cited for citation definitions

namespace slp {

/// Returns the KKT error below which the barrier subproblem for the given
/// barrier parameter is considered solved.
///
/// @tparam Scalar Scalar type.
/// @param μ The barrier parameter.
/// @return The KKT error below which the barrier subproblem for the given
///     barrier parameter is considered solved.
template <typename Scalar>
Scalar barrier_subproblem_tolerance(Scalar μ) {
  // Barrier parameter scale factor for tolerance checks
  constexpr Scalar κ_ε(10);

  return κ_ε * μ;
}

/// Returns the barrier parameter that follows the given one.
///
/// @tparam Scalar Scalar type.
/// @param μ The barrier parameter.
/// @param μ_min The barrier parameter minimum.
/// @return The barrier parameter that follows the given one.
template <typename Scalar>
Scalar next_barrier_parameter(Scalar μ, Scalar μ_min) {
  using std::pow;

  // Barrier parameter linear decrease power in "κ_μ μ". Range of (0, 1).
  constexpr Scalar κ_μ(0.2);

  // Barrier parameter superlinear decrease power in "μ^(θ_μ)". Range of (1, 2).
  constexpr Scalar θ_μ(1.5);

  //   μⱼ₊₁ = max(μₘᵢₙ, min(κ_μ μⱼ, μⱼ^θ_μ))
  //
  // See equation (7) of [2].
  return std::max(μ_min, std::min(κ_μ * μ, pow(μ, θ_μ)));
}

}  // namespace slp
