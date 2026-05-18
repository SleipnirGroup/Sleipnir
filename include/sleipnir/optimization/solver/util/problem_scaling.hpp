// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "sleipnir/optimization/solver/util/sparse_inf_norms.hpp"

// See docs/algorithms.md#Works_cited for citation definitions

namespace slp {

/// Automatic problem scaling factors.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
struct ProblemScaling {
  /// Type alias for dense vector.
  using DenseVector = Eigen::Vector<Scalar, Eigen::Dynamic>;

  /// Cost scaling factor d_f.
  Scalar f = Scalar(1);

  /// Equality constraint scaling factors d_cₑ.
  DenseVector c_e;

  /// Inequality constraint scaling factors d_cᵢ.
  DenseVector c_i;

  /// Computes interior-point problem scaling.
  ///
  /// Scales the objective and each constraint so the largest gradient
  /// component at the starting point is at most gₘₐₓ:
  ///
  ///   d_f    = min(1, gₘₐₓ / ‖∇f(x₀)‖_∞)
  ///   d_c[j] = min(1, gₘₐₓ / ‖∇cⱼ(x₀)‖_∞)
  ///
  /// See §3.8 Automatic Scaling of the Problem Statement in [2].
  ///
  /// @tparam Gradient Cost gradient autodiff type.
  /// @tparam Jacobian Constraint Jacobian autodiff type.
  /// @param g Cost gradient autodiff ∇f, evaluated at the starting point.
  /// @param A_e Equality constraint Jacobian autodiff Aₑ, evaluated at the
  ///            starting point.
  /// @param A_i Inequality constraint Jacobian autodiff Aᵢ, evaluated at the
  ///            starting point.
  /// @return The interior-point problem scaling.
  template <typename Gradient, typename Jacobian>
  static ProblemScaling interior_point(Gradient& g, Jacobian& A_e,
                                       Jacobian& A_i) {
    constexpr Scalar g_max(100);

    const DenseVector grad_f = g.value();
    return ProblemScaling{
        std::min(Scalar(1), g_max / grad_f.template lpNorm<Eigen::Infinity>()),
        (g_max / sparse_inf_norms(A_e.value()).array()).min(Scalar(1)).matrix(),
        (g_max / sparse_inf_norms(A_i.value()).array())
            .min(Scalar(1))
            .matrix()};
  }

  /// Whether the problem scaling is identity.
  ///
  /// @return True if the problem scaling is identity.
  bool is_identity() const {
    return f == Scalar(1) && c_e.size() == 0 && c_i.size() == 0;
  }
};

}  // namespace slp
