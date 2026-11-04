// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace slp {

/// Ruiz equilibration of a symmetric sparse matrix stored as its lower
/// triangle.
///
/// Computes a diagonal S such that every row of S A S has its largest entry
/// close to 1.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class RuizEquilibration {
 public:
  /// Type alias for dense vector.
  using DenseVector = Eigen::VectorX<Scalar>;
  /// Type alias for sparse matrix.
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;

  /// Computes the equilibration of a symmetric matrix.
  ///
  /// @param lower The matrix's lower triangle.
  void compute(const SparseMatrix& lower) {
    using std::abs;

    // Each sweep moves every row's largest entry closer to 1. A few are enough
    // because the pivot tests only need rough scaling.
    constexpr int num_sweeps = 3;

    const int size = lower.rows();
    const Scalar* values = lower.valuePtr();
    const auto* outer = lower.outerIndexPtr();
    const auto* inner = lower.innerIndexPtr();

    m_scale.setOnes(size);
    m_row_max.resize(size);
    for (int sweep = 0; sweep < num_sweeps; ++sweep) {
      m_row_max.setZero();
      for (int col = 0; col < size; ++col) {
        const Scalar col_scale = m_scale[col];
        Scalar col_max{0};
        for (auto k = outer[col]; k < outer[col + 1]; ++k) {
          const int row = inner[k];
          const Scalar value = abs(values[k]) * m_scale[row] * col_scale;
          m_row_max[row] = std::max(m_row_max[row], value);
          col_max = std::max(col_max, value);
        }
        m_row_max[col] = std::max(m_row_max[col], col_max);
      }

      // Empty rows keep their scale
      m_row_max = (m_row_max.array() > Scalar(0)).select(m_row_max, Scalar(1));
      m_scale.array() /= m_row_max.array().sqrt();
    }
  }

  /// Writes S A S into a matrix with the same sparsity pattern as A.
  ///
  /// @param lower A's lower triangle.
  /// @param scaled The lower triangle of S A S. Its sparsity pattern must
  ///     already match lower's.
  void apply(const SparseMatrix& lower, SparseMatrix& scaled) const {
    const Scalar* values = lower.valuePtr();
    const auto* outer = lower.outerIndexPtr();
    const auto* inner = lower.innerIndexPtr();
    Scalar* scaled_values = scaled.valuePtr();

    for (int col = 0; col < lower.cols(); ++col) {
      const Scalar col_scale = m_scale[col];
      for (auto k = outer[col]; k < outer[col + 1]; ++k) {
        scaled_values[k] = values[k] * (m_scale[inner[k]] * col_scale);
      }
    }
  }

  /// Returns the diagonal of S.
  ///
  /// @return The diagonal of S.
  const DenseVector& scale() const { return m_scale; }

 private:
  /// The diagonal of S.
  DenseVector m_scale;

  /// Each row's largest entry during a sweep, kept to avoid reallocating.
  DenseVector m_row_max;
};

}  // namespace slp
