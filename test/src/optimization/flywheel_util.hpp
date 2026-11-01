// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>

/// A flywheel state and input trajectory.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct FlywheelTrajectory {
  /// States x₀, …, x_N.
  Eigen::RowVectorX<Scalar> X;

  /// Inputs u₀, …, u_N₋₁.
  Eigen::RowVectorX<Scalar> U;
};

/// Returns the trajectory that minimizes Σ(r − xₖ)² for k = 0, …, N subject to
/// xₖ₊₁ = Axₖ + Buₖ, x₀ = 0, and |uₖ| ≤ u_max, where 0 < A < 1, B > 0, r > 0,
/// and u_max ≥ (1 − A)r/B (the input that holds xₖ = r).
///
/// @tparam Scalar Scalar type.
/// @param A Discrete state transition.
/// @param B Discrete input gain.
/// @param r Reference state.
/// @param u_max Input bound.
/// @param N Number of timesteps.
template <typename Scalar>
FlywheelTrajectory<Scalar> optimal_flywheel_trajectory(Scalar A, Scalar B,
                                                       Scalar r, Scalar u_max,
                                                       int N) {
  // Every xₖ increases with each of u₀, …, uₖ₋₁, so each cost term is minimized
  // by driving xₖ as close to r as the input bound allows. Saturating the input
  // until one step can land exactly on r, then holding it there, minimizes
  // every term at once.
  //
  // Once xₖ = r, the landing input is uₖ = (r − Ar)/B = (1 − A)r/B, the
  // steady-state input, so the landing branch also holds the reference.

  FlywheelTrajectory<Scalar> trajectory{Eigen::RowVectorX<Scalar>(N + 1),
                                        Eigen::RowVectorX<Scalar>(N)};
  trajectory.X(0) = Scalar(0);

  for (int k = 0; k < N; ++k) {
    Scalar x = trajectory.X(k);

    if (A * x + B * u_max < r) {
      // Saturate the input until the reference is within one step
      trajectory.U(k) = u_max;
      trajectory.X(k + 1) = A * x + B * u_max;
    } else {
      // Land exactly on (or hold) the reference
      trajectory.U(k) = (r - A * x) / B;
      trajectory.X(k + 1) = r;
    }
  }

  return trajectory;
}
