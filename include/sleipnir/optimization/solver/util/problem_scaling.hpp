// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>

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
  DenseVector c_e = DenseVector{};

  /// Inequality constraint scaling factors d_cᵢ.
  DenseVector c_i = DenseVector{};

  /// Whether the problem scaling is identity.
  ///
  /// @return True if the problem scaling is identity.
  bool is_identity() const {
    return f == Scalar(1) && c_e.size() == 0 && c_i.size() == 0;
  }
};

}  // namespace slp
