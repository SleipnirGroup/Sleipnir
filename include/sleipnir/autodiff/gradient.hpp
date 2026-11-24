// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <utility>

#include <Eigen/SparseCore>

#include "sleipnir/autodiff/jacobian.hpp"
#include "sleipnir/autodiff/matrix.hpp"
#include "sleipnir/autodiff/real.hpp"
#include "sleipnir/util/concepts.hpp"
#include "sleipnir/util/symbol_exports.hpp"

namespace slp {

/// This class calculates the gradient of an expression with respect to a vector
/// of variables.
///
/// The gradient is only recomputed if the expression is quadratic or higher
/// order.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class Gradient {
 public:
  /// Constructs a Gradient object.
  ///
  /// @param expression Expression of which to compute the gradient.
  /// @param wrt Variable with respect to which to compute the gradient.
  Gradient(Real<Scalar> expression, Real<Scalar> wrt)
      : m_jacobian{std::move(expression), std::move(wrt)} {}

  /// Constructs a Gradient object.
  ///
  /// @param expression Expression of which to compute the gradient.
  /// @param wrt Vector of variables with respect to which to compute the
  ///     gradient.
  Gradient(Real<Scalar> expression, SleipnirMatrixLike<Scalar> auto wrt)
      : m_jacobian{Matrix{std::move(expression)}, std::move(wrt)} {}

  /// Returns the gradient as a Matrix.
  ///
  /// This is useful when constructing optimization problems with derivatives in
  /// them.
  ///
  /// @return The gradient as a Matrix.
  Matrix<Scalar> get() const { return m_jacobian.get().T(); }

  /// Evaluates the gradient at wrt's value.
  ///
  /// @return The gradient at wrt's value.
  const Eigen::SparseVector<Scalar>& value() {
    const auto& J = m_jacobian.value();

    if (m_g.rows() == 0) {
      m_g = J.transpose();
    } else {
      // J's sparsity pattern is constant, so m_g's indices from the first
      // transpose are still valid. J is column-major with one row, so each
      // column holds at most one value and the values are sorted by column
      // index. This matches m_g's index order, so we can copy the values
      // directly.
      std::copy_n(J.valuePtr(), J.nonZeros(), m_g.valuePtr());
    }

    return m_g;
  }

 private:
  Eigen::SparseVector<Scalar> m_g;

  Jacobian<Scalar> m_jacobian;
};

extern template class EXPORT_TEMPLATE_DECLARE(SLEIPNIR_DLLEXPORT)
Gradient<double>;

}  // namespace slp
