// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cstddef>

#include <Eigen/Core>
#include <Eigen/OrderingMethods>
#include <Eigen/SparseCore>
#include <gch/small_vector.hpp>

namespace slp {

/// Fill-reducing ordering of a KKT matrix
///
///   [H   Aₑᵀ]
///   [Aₑ  −γI]
///
/// for LDLT factorization without pivoting.
///
/// The ordering is AMD's, except each equality constraint row is eliminated
/// after at least one of the decision variables it involves. Otherwise, a
/// constraint row eliminated first has a pivot of exactly −γ, and the decision
/// variables it involves pick up terms of order 1/γ. That element growth ruins
/// the factorization's accuracy unless γ is too large for accurate steps, which
/// matters most with low precision Scalar types.
///
/// Eigen's ordering methods only see the matrix, and the sparsity pattern
/// doesn't say which rows are constraints, so this computes the permutation
/// itself and permutes the matrix for a solver that uses it as given.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class KKTOrdering {
 public:
  /// Type alias for sparse matrix.
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  /// Type alias for sparse matrix storage index.
  using StorageIndex = typename SparseMatrix::StorageIndex;
  /// Type alias for permutation matrix.
  using Permutation =
      Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, StorageIndex>;

  /// Computes the ordering of a KKT matrix.
  ///
  /// @param lhs The KKT matrix's lower triangle. It must be compressed.
  /// @param num_decision_variables The number of decision variables, which
  ///     come before the equality constraints in the KKT matrix's rows.
  void analyze(const SparseMatrix& lhs, int num_decision_variables) {
    compute_ordering(lhs, num_decision_variables);
    compute_permuted_pattern(lhs);
  }

  /// Returns the upper triangle of P lhs Pᵀ, where P is the permutation.
  ///
  /// @param lhs The KKT matrix's lower triangle. It must be compressed and have
  ///     the sparsity pattern of the matrix given to analyze().
  /// @return The upper triangle of P lhs Pᵀ.
  const SparseMatrix& permute(const SparseMatrix& lhs) {
    for (size_t k = 0; k < m_permuted_value_indices.size(); ++k) {
      m_permuted_lhs.valuePtr()[m_permuted_value_indices[k]] =
          lhs.valuePtr()[k];
    }

    return m_permuted_lhs;
  }

  /// Returns the permutation P. Row i of the KKT matrix is row P(i) of the
  /// permuted matrix.
  ///
  /// @return The permutation P.
  const Permutation& permutation() const { return m_P; }

  /// Returns the inverse permutation P⁻¹. Row i of the permuted matrix is row
  /// P⁻¹(i) of the KKT matrix, so P⁻¹ lists the KKT matrix's rows in
  /// elimination order.
  ///
  /// @return The inverse permutation P⁻¹.
  const Permutation& inverse_permutation() const { return m_P_inv; }

 private:
  /// The upper triangle of the most recently permuted KKT matrix.
  SparseMatrix m_permuted_lhs;

  /// The index in m_permuted_lhs's values of each of the KKT matrix's values.
  gch::small_vector<StorageIndex> m_permuted_value_indices;

  /// The permutation P.
  Permutation m_P;

  /// The inverse permutation P⁻¹.
  Permutation m_P_inv;

  /// Computes the permutation.
  ///
  /// @param lhs The KKT matrix's lower triangle.
  /// @param num_decision_variables The number of decision variables.
  void compute_ordering(const SparseMatrix& lhs, int num_decision_variables) {
    Eigen::AMDOrdering<StorageIndex>{}(
        lhs.template selfadjointView<Eigen::Lower>(), m_P_inv);

    // In the lower triangle, a decision variable's column holds the constraint
    // rows that involve it
    const int num_rows = lhs.rows();
    gch::small_vector<int> num_variables(num_rows, 0);
    for (int col = 0; col < num_decision_variables; ++col) {
      for (typename SparseMatrix::InnerIterator it{lhs, col}; it; ++it) {
        if (it.row() >= num_decision_variables) {
          ++num_variables[it.row()];
        }
      }
    }

    // Whether each constraint row involves a placed decision variable
    gch::small_vector<bool> has_placed_variable(num_rows, false);
    gch::small_vector<bool> deferred(num_rows, false);
    gch::small_vector<bool> placed(num_rows, false);
    gch::small_vector<StorageIndex> order;
    order.reserve(num_rows);

    auto place = [&](int row) {
      placed[row] = true;
      order.emplace_back(row);
      if (row >= num_decision_variables) {
        return;
      }

      // Place the deferred constraint rows that were waiting on this decision
      // variable
      for (typename SparseMatrix::InnerIterator it{lhs, row}; it; ++it) {
        int constraint_row = it.row();
        if (constraint_row < num_decision_variables) {
          continue;
        }
        has_placed_variable[constraint_row] = true;
        if (deferred[constraint_row] && !placed[constraint_row]) {
          placed[constraint_row] = true;
          order.emplace_back(constraint_row);
        }
      }
    };

    for (int i = 0; i < num_rows; ++i) {
      int row = m_P_inv.indices()(i);
      if (placed[row]) {
        continue;
      }

      if (row >= num_decision_variables && !has_placed_variable[row] &&
          num_variables[row] > 0) {
        deferred[row] = true;
      } else {
        place(row);
      }
    }

    for (int i = 0; i < num_rows; ++i) {
      m_P_inv.indices()(i) = order[i];
    }
    m_P = m_P_inv.inverse();
  }

  /// Computes the sparsity pattern of the permuted matrix and where each of the
  /// KKT matrix's values goes in it.
  ///
  /// All KKT matrices share a sparsity pattern, so permuting one only has to
  /// copy values instead of rebuilding the whole matrix.
  ///
  /// @param lhs The KKT matrix's lower triangle.
  void compute_permuted_pattern(const SparseMatrix& lhs) {
    // Entry (row, col) of the lower triangle moves to (P(row), P(col)), which
    // the upper triangle stores in column max(P(row), P(col)) at row
    // min(P(row), P(col))
    auto permuted_col = [&](StorageIndex row, StorageIndex col) {
      return std::max(m_P.indices()(row), m_P.indices()(col));
    };
    auto permuted_row = [&](StorageIndex row, StorageIndex col) {
      return std::min(m_P.indices()(row), m_P.indices()(col));
    };

    const int num_rows = lhs.rows();
    m_permuted_lhs.resize(num_rows, num_rows);
    m_permuted_lhs.resizeNonZeros(lhs.nonZeros());

    // Count each permuted column's entries, then turn the counts into column
    // start offsets
    StorageIndex* outer = m_permuted_lhs.outerIndexPtr();
    std::fill(outer, outer + num_rows + 1, StorageIndex{0});
    for (int col = 0; col < num_rows; ++col) {
      for (typename SparseMatrix::InnerIterator it{lhs, col}; it; ++it) {
        ++outer[permuted_col(it.row(), col) + 1];
      }
    }
    for (int col = 0; col < num_rows; ++col) {
      outer[col + 1] += outer[col];
    }

    // Assign each entry the next free slot in its permuted column. lhs is
    // compressed, so iterating over its columns visits its values in storage
    // order.
    gch::small_vector<StorageIndex> next(outer, outer + num_rows);
    m_permuted_value_indices.clear();
    m_permuted_value_indices.reserve(lhs.nonZeros());
    for (int col = 0; col < num_rows; ++col) {
      for (typename SparseMatrix::InnerIterator it{lhs, col}; it; ++it) {
        StorageIndex k = next[permuted_col(it.row(), col)]++;
        m_permuted_lhs.innerIndexPtr()[k] = permuted_row(it.row(), col);
        m_permuted_value_indices.emplace_back(k);
      }
    }
  }
};

}  // namespace slp
