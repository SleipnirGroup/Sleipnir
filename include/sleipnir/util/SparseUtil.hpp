// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <concepts>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "sleipnir/SymbolExports.hpp"

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

/**
 * Write the sparsity pattern of a sparse matrix to a file.
 *
 * Each character represents an element with "." representing zero, "+"
 * representing positive, and "-" representing negative. Here's an example for a
 * 3x3 identity matrix.
 *
 * "+.."
 * ".+."
 * "..+"
 *
 * @param[in] filename The filename.
 * @param[in] mat The sparse matrix.
 */
SLEIPNIR_DLLEXPORT void Spy(std::string_view filename,
                            const Eigen::SparseMatrix<double>& mat);

/**
 * Write the sparsity pattern of a sparse matrix to a file.
 *
 * Each character represents an element with '.' representing zero, '+'
 * representing positive, and '-' representing negative. Here's an example for a
 * 3x3 identity matrix.
 *
 * "+.."
 * ".+."
 * "..+"
 *
 * @param[out] file A file stream.
 * @param[in] mat The sparse matrix.
 */
SLEIPNIR_DLLEXPORT void Spy(std::ostream& file,
                            const Eigen::SparseMatrix<double>& mat);

}  // namespace sleipnir
