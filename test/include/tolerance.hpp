// Copyright (c) Sleipnir contributors

#pragma once

#include <cmath>
#include <limits>

#include <sleipnir/optimization/solver/options.hpp>

// Test tolerances follow the error model of the value being checked, so they
// scale correctly with Scalar's precision. See "Choosing test tolerances" in
// docs/contributing.md.

/// Returns the tolerance for a value whose error is dominated by rounding
/// error, a multiple of ε.
///
/// Use this for values computed without the solver (e.g., autodiff
/// derivatives). Values the solver produces are only as accurate as its
/// termination tolerance, so use solution_tolerance() for those, even if their
/// error is usually much smaller.
///
/// @tparam Scalar Scalar type.
/// @param multiple Error bound in multiples of machine epsilon.
template <typename Scalar>
constexpr Scalar rounding_tolerance(double multiple) {
  return Scalar(multiple) * std::numeric_limits<Scalar>::epsilon();
}

/// How a solution's error depends on the solver's termination tolerance tol.
///
/// The solver stops once the KKT error is below tol.
enum class SolutionType {
  /// LICQ, strict complementarity, and second-order sufficiency hold, so the
  /// solution error is linear in the KKT error, O(tol).
  REGULAR,
  /// One of those fails, so the solution error is O(√tol). For example, a
  /// constraint is weakly active (active with a zero Lagrange multiplier),
  /// there are no finite Lagrange multipliers, or the cost is nearly flat in a
  /// feasible direction (e.g., an input mode that barely affects the cost).
  DEGENERATE
};

/// Returns the tolerance for a value whose error is bounded by the solver's
/// termination tolerance, a multiple of tol for a regular solution or √tol for
/// a degenerate solution.
///
/// tol is the default solver tolerance, √ε. For a regular solution, the
/// multiple is the sensitivity of the value to the KKT error.
///
/// Equality constraint residuals (e.g., initial and final states, and dynamics
/// defects) are part of the KKT error, so the solver guarantees they're at most
/// tol. Use a multiple of at least 1 for them; their error is often much
/// smaller, but how much smaller depends on the iterates' path, which varies
/// across platforms.
///
/// @tparam Scalar Scalar type.
/// @param multiple Error bound in multiples of tol or √tol.
/// @param type How the solution error depends on tol.
template <typename Scalar>
Scalar solution_tolerance(double multiple,
                          SolutionType type = SolutionType::REGULAR) {
  using std::sqrt;
  Scalar tolerance = slp::Options<Scalar>{}.tolerance;
  if (type == SolutionType::DEGENERATE) {
    tolerance = sqrt(tolerance);
  }
  return Scalar(multiple) * tolerance;
}
