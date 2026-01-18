// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "sleipnir/optimization/solver/ipm_matrix_callbacks.hpp"
#include "sleipnir/optimization/solver/util/barrier_parameter.hpp"

namespace slp {

/// Sets the interior-point method's slack variables, dual variables, and
/// barrier parameter to a cold start for the given initial guess.
///
/// @tparam Scalar Scalar type.
/// @param[in] matrices Matrix callbacks.
/// @param[in] x The initial guess for the decision variables.
/// @param[out] s The inequality constraint slack variables.
/// @param[out] y The equality constraint dual variables.
/// @param[out] z The inequality constraint dual variables.
/// @param[out] μ The barrier parameter.
template <typename Scalar>
void set_ipm_cold_start_initial_iterate(
    const IPMMatrixCallbacks<Scalar>& matrices, const Eigen::VectorX<Scalar>& x,
    Eigen::VectorX<Scalar>& s, Eigen::VectorX<Scalar>& y,
    Eigen::VectorX<Scalar>& z, Scalar& μ) {
  using DenseVector = Eigen::VectorX<Scalar>;

  // If cᵢ(x) is sufficiently feasible, start the constraint with zero
  // violation. Otherwise, make Σ = S⁻¹Z ≈ I.
  const DenseVector c_i = matrices.c_i(x);
  s = (c_i.array() >= Scalar(1e-2)).select(c_i, Scalar(1));

  y = DenseVector::Zero(matrices.num_equality_constraints);
  z = DenseVector::Ones(matrices.num_inequality_constraints);
  μ = Scalar(0.1) * matrices.scaling.f;
}

/// The result of moving a previous solve's iterate back along the central path.
struct IPMWarmStartResult {
  /// False if the previous solve's iterate is no better a starting point than a
  /// cold start, in which case the caller should use a cold start instead.
  bool recentered;

  /// Whether the inequality constraint dual variables changed.
  bool z_changed;
};

/// Moves a previous solve's slack variables, dual variables, and barrier
/// parameter back along the central path to suit a new initial guess.
///
/// The barrier parameter becomes the one a cold start would have reached with
/// an iterate of the same KKT error, but no smaller than the previous solve's.
/// Slack-dual pairs below the new central path are moved onto it.
///
/// @tparam Scalar Scalar type.
/// @param[in] matrices Matrix callbacks.
/// @param[in] g The cost function gradient at the initial guess.
/// @param[in] A_e The equality constraint Jacobian at the initial guess.
/// @param[in] c_e The equality constraints at the initial guess.
/// @param[in] A_i The inequality constraint Jacobian at the initial guess.
/// @param[in] c_i The inequality constraints at the initial guess.
/// @param[in] include_equality_constraints Whether the equality constraint
///     violation counts toward the iterate's KKT error.
/// @param[in,out] s The previous solve's inequality constraint slack variables.
/// @param[in] y The previous solve's equality constraint dual variables.
/// @param[in,out] z The previous solve's inequality constraint dual variables.
/// @param[in,out] μ The previous solve's barrier parameter.
/// @return Whether the iterate was recentered and whether z changed.
template <typename Scalar>
IPMWarmStartResult set_ipm_warm_start_initial_iterate(
    const IPMMatrixCallbacks<Scalar>& matrices,
    const Eigen::SparseVector<Scalar>& g,
    const Eigen::SparseMatrix<Scalar>& A_e, const Eigen::VectorX<Scalar>& c_e,
    const Eigen::SparseMatrix<Scalar>& A_i, const Eigen::VectorX<Scalar>& c_i,
    bool include_equality_constraints, Eigen::VectorX<Scalar>& s,
    const Eigen::VectorX<Scalar>& y, Eigen::VectorX<Scalar>& z, Scalar& μ) {
  using DenseVector = Eigen::VectorX<Scalar>;

  const Scalar prev_μ = μ;
  const Scalar μ_cold = Scalar(0.1) * matrices.scaling.f;

  // Raise the slacks of constraints that are more feasible at the new initial
  // guess, which reduces their constraint violation cᵢ − s
  s = s.cwiseMax(c_i);

  // The KKT error for barrier parameter μ.
  //
  //   max(‖∇f − Aₑᵀy − Aᵢᵀz‖_∞, ‖Sz − μe‖_∞, ‖cₑ‖_∞, ‖cᵢ − s‖_∞)
  //
  // where ‖cₑ‖_∞ is only included if requested.
  //
  // Unlike the solver's termination criteria, the stationarity and
  // complementarity residuals aren't scaled down by the size of the duals. The
  // scaling makes an iterate with diverged duals look nearly optimal.
  //
  // E_fixed holds the residuals that don't depend on μ.
  const DenseVector stationarity =
      g - A_e.transpose() * y - A_i.transpose() * z;
  Scalar E_fixed = std::max({stationarity.template lpNorm<Eigen::Infinity>(),
                             (c_i - s).template lpNorm<Eigen::Infinity>()});
  if (include_equality_constraints) {
    E_fixed = std::max(E_fixed, c_e.template lpNorm<Eigen::Infinity>());
  }
  const DenseVector Sz = s.cwiseProduct(z);
  auto E_μ = [&](Scalar μ) {
    return std::max(
        E_fixed, (Sz.array() - μ).matrix().template lpNorm<Eigen::Infinity>());
  };

  μ = μ_cold;
  while (μ > prev_μ && E_μ(μ) <= barrier_subproblem_tolerance(μ)) {
    μ = next_barrier_parameter(μ, prev_μ);
  }

  // If the iterate doesn't solve the first barrier subproblem, it's no better
  // than a cold start
  if (μ >= μ_cold) {
    return {.recentered = false, .z_changed = false};
  }

  bool z_changed = false;

  // If the barrier parameter increased, slack-dual pairs with sᵢzᵢ < μ are
  // closer to their bounds than the central path for μ is. Move each back to
  // the central path sᵢzᵢ = μ by raising the smaller of the two, which is the
  // one closest to its bound.
  if (μ > prev_μ) {
    for (int row = 0; row < s.rows(); ++row) {
      if (s[row] * z[row] < μ) {
        if (s[row] < z[row]) {
          s[row] = μ / z[row];
        } else {
          z[row] = μ / s[row];
          z_changed = true;
        }
      }
    }
  }

  return {.recentered = true, .z_changed = z_changed};
}

}  // namespace slp
