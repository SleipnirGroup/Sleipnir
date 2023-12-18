// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "sleipnir/util/SymbolExports.hpp"

namespace sleipnir {

/**
 * Converts dense column vector into sparse diagonal matrix.
 *
 * @param src Column vector.
 */
SLEIPNIR_DLLEXPORT Eigen::SparseMatrix<double> SparseDiagonal(
    const Eigen::VectorXd& src);

/**
 * Returns sparse identity matrix.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 */
SLEIPNIR_DLLEXPORT Eigen::SparseMatrix<double> SparseIdentity(int rows,
                                                              int cols);

/**
 * Returns lₚ-norm of a sparse matrix.
 *
 * @tparam Power Power of the norm (e.g., 2-norm). Eigen::Infinity refers to the
 *   ∞-norm.
 */
template <int Power, typename Derived>
  requires std::derived_from<Derived, Eigen::SparseCompressedBase<Derived>>
double SparseLpNorm(const Derived& mat) {
  double norm = 0.0;

  for (int k = 0; k < mat.outerSize(); ++k) {
    for (typename Derived::InnerIterator it{mat, k}; it; ++it) {
      if constexpr (Power == 1) {
        norm += std::abs(it.value());
      } else if constexpr (Power == 2) {
        norm += it.value() * it.value();
      } else if constexpr (Power == Eigen::Infinity) {
        norm = std::max(std::abs(norm), std::abs(it.value()));
      } else {
        norm += std::pow(std::abs(it.value()), Power);
      }
    }
  }

  if constexpr (Power == 1 || Power == Eigen::Infinity) {
    return norm;
  } else if constexpr (Power == 2) {
    return std::sqrt(norm);
  } else {
    return std::pow(norm, 1.0 / Power);
  }
}

}  // namespace sleipnir
