// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cassert>
#include <vector>

#include <Eigen/SparseCore>

namespace sleipnir {

/**
 * A submatrix of a sparse matrix builder with reference semantics.
 *
 * @tparam Scalar Matrix scalar type.
 */
template <typename Scalar>
class SparseMatrixBuilderBlock {
 public:
  /**
   * Constructs a sparse matrix block.
   *
   * @param triplets Triplets from sparse matrix to reference.
   * @param rowOffset The row offset of the block selection.
   * @param colOffset The column offset of the block selection.
   * @param blockRows The number of rows in the block selection.
   * @param blockCols The number of columns in the block selection.
   */
  SparseMatrixBuilderBlock(std::vector<Eigen::Triplet<Scalar>>& triplets,
                           int rowOffset, int colOffset, int blockRows,
                           int blockCols)
      : m_triplets{triplets},
        m_rowOffset{rowOffset},
        m_colOffset{colOffset},
        m_blockRows{blockRows},
        m_blockCols{blockCols} {}

  /**
   * Assigns a sparse matrix to the block.
   */
  SparseMatrixBuilderBlock<Scalar>& operator=(
      const Eigen::SparseMatrix<Scalar>& mat) {
    assert(Rows() == mat.rows());
    assert(Cols() == mat.cols());

    for (int k = 0; k < mat.outerSize(); ++k) {
      for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it{mat, k}; it;
           ++it) {
        m_triplets.emplace_back(m_rowOffset + it.row(), m_colOffset + it.col(),
                                it.value());
      }
    }

    return *this;
  }

  /**
   * Returns number of rows in the block.
   */
  int Rows() const { return m_blockRows; }

  /**
   * Returns number of columns in the block.
   */
  int Cols() const { return m_blockCols; }

  /**
   * Set builder to identity matrix.
   */
  void SetIdentity() {
    for (int row = 0; row < std::min(Rows(), Cols()); ++row) {
      m_triplets.emplace_back(m_rowOffset + row, m_colOffset + row, 1.0);
    }
  }

 private:
  std::vector<Eigen::Triplet<Scalar>>& m_triplets;
  int m_rowOffset = 0;
  int m_colOffset = 0;
  int m_blockRows = 0;
  int m_blockCols = 0;
};

/**
 * Builds a sparse matrix from triplets.
 *
 * @tparam Scalar Matrix scalar type.
 */
template <typename Scalar>
class SparseMatrixBuilder {
 public:
  SparseMatrixBuilder(int rows, int cols) : m_rows{rows}, m_cols{cols} {}

  /**
   * Assigns a sparse matrix to the builder.
   */
  SparseMatrixBuilder<Scalar> operator=(
      const Eigen::SparseMatrix<Scalar>& mat) {
    assert(Rows() == mat.rows());
    assert(Cols() == mat.cols());

    for (int k = 0; k < mat.outerSize(); ++k) {
      for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it{mat, k}; it;
           ++it) {
        m_triplets.emplace_back(it.row(), it.col(), it.value());
      }
    }
  }

  /**
   * Returns number of rows in the matrix.
   */
  int Rows() const { return m_rows; }

  /**
   * Returns number of columns in the matrix.
   */
  int Cols() const { return m_cols; }

  /**
   * Returns a block slice of the sparse matrix builder.
   *
   * @param rowOffset The row offset of the block selection.
   * @param colOffset The column offset of the block selection.
   * @param blockRows The number of rows in the block selection.
   * @param blockCols The number of columns in the block selection.
   */
  SparseMatrixBuilderBlock<Scalar> Block(int rowOffset, int colOffset,
                                         int blockRows, int blockCols) {
    assert(rowOffset >= 0 && rowOffset <= Rows());
    assert(colOffset >= 0 && colOffset <= Cols());
    assert(blockRows >= 0 && blockRows <= Rows() - rowOffset);
    assert(blockCols >= 0 && blockCols <= Cols() - colOffset);
    return SparseMatrixBuilderBlock<Scalar>{m_triplets, rowOffset, colOffset,
                                            blockRows, blockCols};
  }

  /**
   * Set builder to identity matrix.
   */
  void SetIdentity() {
    for (int row = 0; row < std::min(Rows(), Cols()); ++row) {
      m_triplets.emplace_back(row, row, 1.0);
    }
  }

  /**
   * Creates a sparse matrix from the list of provided triplets.
   */
  Eigen::SparseMatrix<Scalar> Build() const {
    Eigen::SparseMatrix<Scalar> mat{Rows(), Cols()};
    mat.setFromTriplets(m_triplets.begin(), m_triplets.end());
    return mat;
  }

  /**
   * Clears the internal triplets.
   */
  void Clear() { m_triplets.clear(); }

 private:
  std::vector<Eigen::Triplet<Scalar>> m_triplets;
  int m_rows = 0;
  int m_cols = 0;
};

}  // namespace sleipnir
