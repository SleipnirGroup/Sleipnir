// Copyright (c) Sleipnir contributors

#pragma once

#include <cassert>
#include <concepts>
#include <initializer_list>
#include <iterator>
#include <span>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "sleipnir/autodiff/Variable.hpp"
#include "sleipnir/autodiff/VariableBlock.hpp"
#include "sleipnir/util/SymbolExports.hpp"

namespace sleipnir {

/**
 * A matrix of autodiff variables.
 */
class SLEIPNIR_DLLEXPORT VariableMatrix {
 public:
  /**
   * Constructs an empty VariableMatrix.
   */
  VariableMatrix() = default;

  /**
   * Constructs a VariableMatrix column vector with the given rows.
   *
   * @param rows The number of matrix rows.
   */
  explicit VariableMatrix(int rows);

  /**
   * Constructs a VariableMatrix with the given dimensions.
   *
   * @param rows The number of matrix rows.
   * @param cols The number of matrix columns.
   */
  VariableMatrix(int rows, int cols);

  /**
   * Constructs a scalar VariableMatrix from a nested list of Variables.
   *
   * @param list The nested list of Variables.
   */
  VariableMatrix(
      std::initializer_list<std::initializer_list<Variable>> list);  // NOLINT

  /**
   * Constructs a scalar VariableMatrix from a nested list of doubles.
   *
   * This overload is for Python bindings only.
   *
   * @param list The nested list of Variables.
   */
  VariableMatrix(std::vector<std::vector<double>> list);  // NOLINT

  /**
   * Constructs a scalar VariableMatrix from a nested list of Variables.
   *
   * This overload is for Python bindings only.
   *
   * @param list The nested list of Variables.
   */
  VariableMatrix(std::vector<std::vector<Variable>> list);  // NOLINT

  /**
   * Constructs a VariableMatrix from an Eigen matrix.
   */
  template <typename Derived>
  VariableMatrix(const Eigen::MatrixBase<Derived>& values)  // NOLINT
      : m_rows{static_cast<int>(values.rows())},
        m_cols{static_cast<int>(values.cols())} {
    m_storage.reserve(values.rows() * values.cols());
    for (int row = 0; row < values.rows(); ++row) {
      for (int col = 0; col < values.cols(); ++col) {
        m_storage.emplace_back(values(row, col));
      }
    }
  }

  /**
   * Constructs a VariableMatrix from an Eigen diagonal matrix.
   */
  template <typename Derived>
  VariableMatrix(const Eigen::DiagonalBase<Derived>& values)  // NOLINT
      : m_rows{static_cast<int>(values.rows())},
        m_cols{static_cast<int>(values.cols())} {
    m_storage.reserve(values.rows() * values.cols());
    for (int row = 0; row < values.rows(); ++row) {
      for (int col = 0; col < values.cols(); ++col) {
        if (row == col) {
          m_storage.emplace_back(values.diagonal()(row));
        } else {
          m_storage.emplace_back(0.0);
        }
      }
    }
  }

  /**
   * Assigns an Eigen matrix to a VariableMatrix.
   */
  template <typename Derived>
  VariableMatrix& operator=(const Eigen::MatrixBase<Derived>& values) {
    assert(Rows() == values.rows());
    assert(Cols() == values.cols());

    for (int row = 0; row < values.rows(); ++row) {
      for (int col = 0; col < values.cols(); ++col) {
        (*this)(row, col) = values(row, col);
      }
    }

    return *this;
  }

  /**
   * Sets the VariableMatrix's internal values.
   */
  template <typename Derived>
    requires std::same_as<typename Derived::Scalar, double>
  VariableMatrix& SetValue(const Eigen::MatrixBase<Derived>& values) {
    assert(Rows() == values.rows());
    assert(Cols() == values.cols());

    for (int row = 0; row < values.rows(); ++row) {
      for (int col = 0; col < values.cols(); ++col) {
        (*this)(row, col).SetValue(values(row, col));
      }
    }

    return *this;
  }

  /**
   * Constructs a scalar VariableMatrix from a Variable.
   */
  VariableMatrix(const Variable& variable);  // NOLINT

  /**
   * Constructs a scalar VariableMatrix from a Variable.
   */
  VariableMatrix(Variable&& variable);  // NOLINT

  /**
   * Constructs a VariableMatrix from a VariableBlock.
   */
  VariableMatrix(const VariableBlock<VariableMatrix>& values);  // NOLINT

  /**
   * Constructs a VariableMatrix from a VariableBlock.
   */
  VariableMatrix(const VariableBlock<const VariableMatrix>& values);  // NOLINT

  /**
   * Constructs a column vector wrapper around a Variable array.
   *
   * @param values Variable array to wrap.
   */
  explicit VariableMatrix(std::span<Variable> values);

  /**
   * Constructs a matrix wrapper around a Variable array.
   *
   * @param values Variable array to wrap.
   * @param rows The number of matrix rows.
   * @param cols The number of matrix columns.
   */
  VariableMatrix(std::span<Variable> values, int rows, int cols);

  /**
   * Returns a block pointing to the given row and column.
   *
   * @param row The block row.
   * @param col The block column.
   */
  Variable& operator()(int row, int col);

  /**
   * Returns a block pointing to the given row and column.
   *
   * @param row The block row.
   * @param col The block column.
   */
  const Variable& operator()(int row, int col) const;

  /**
   * Returns a block pointing to the given row.
   *
   * @param row The block row.
   */
  Variable& operator()(int row);

  /**
   * Returns a block pointing to the given row.
   *
   * @param row The block row.
   */
  const Variable& operator()(int row) const;

  /**
   * Returns a block slice of the variable matrix.
   *
   * @param rowOffset The row offset of the block selection.
   * @param colOffset The column offset of the block selection.
   * @param blockRows The number of rows in the block selection.
   * @param blockCols The number of columns in the block selection.
   */
  VariableBlock<VariableMatrix> Block(int rowOffset, int colOffset,
                                      int blockRows, int blockCols);

  /**
   * Returns a block slice of the variable matrix.
   *
   * @param rowOffset The row offset of the block selection.
   * @param colOffset The column offset of the block selection.
   * @param blockRows The number of rows in the block selection.
   * @param blockCols The number of columns in the block selection.
   */
  const VariableBlock<const VariableMatrix> Block(int rowOffset, int colOffset,
                                                  int blockRows,
                                                  int blockCols) const;

  /**
   * Returns a segment of the variable vector.
   *
   * @param offset The offset of the segment.
   * @param length The length of the segment.
   */
  VariableBlock<VariableMatrix> Segment(int offset, int length);

  /**
   * Returns a segment of the variable vector.
   *
   * @param offset The offset of the segment.
   * @param length The length of the segment.
   */
  const VariableBlock<const VariableMatrix> Segment(int offset,
                                                    int length) const;

  /**
   * Returns a row slice of the variable matrix.
   *
   * @param row The row to slice.
   */
  VariableBlock<VariableMatrix> Row(int row);

  /**
   * Returns a row slice of the variable matrix.
   *
   * @param row The row to slice.
   */
  const VariableBlock<const VariableMatrix> Row(int row) const;

  /**
   * Returns a column slice of the variable matrix.
   *
   * @param col The column to slice.
   */
  VariableBlock<VariableMatrix> Col(int col);

  /**
   * Returns a column slice of the variable matrix.
   *
   * @param col The column to slice.
   */
  const VariableBlock<const VariableMatrix> Col(int col) const;

  /**
   * Matrix multiplication operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT VariableMatrix operator*(const VariableMatrix& lhs,
                                                     const VariableMatrix& rhs);

  /**
   * Matrix-scalar multiplication operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT VariableMatrix operator*(const VariableMatrix& lhs,
                                                     const Variable& rhs);

  /**
   * Matrix-scalar multiplication operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT VariableMatrix operator*(const VariableMatrix& lhs,
                                                     double rhs);

  /**
   * Scalar-matrix multiplication operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT VariableMatrix operator*(const Variable& lhs,
                                                     const VariableMatrix& rhs);

  /**
   * Scalar-matrix multiplication operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT VariableMatrix operator*(double lhs,
                                                     const VariableMatrix& rhs);

  /**
   * Compound matrix multiplication-assignment operator.
   *
   * @param rhs Variable to multiply.
   */
  VariableMatrix& operator*=(const VariableMatrix& rhs);

  /**
   * Binary division operator (only enabled when rhs is a scalar).
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT VariableMatrix operator/(const VariableMatrix& lhs,
                                                     const Variable& rhs);

  /**
   * Compound matrix division-assignment operator (only enabled when rhs
   * is a scalar).
   *
   * @param rhs Variable to divide.
   */
  VariableMatrix& operator/=(const Variable& rhs);

  /**
   * Binary addition operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT VariableMatrix operator+(const VariableMatrix& lhs,
                                                     const VariableMatrix& rhs);

  /**
   * Compound addition-assignment operator.
   *
   * @param rhs Variable to add.
   */
  VariableMatrix& operator+=(const VariableMatrix& rhs);

  /**
   * Binary subtraction operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT VariableMatrix operator-(const VariableMatrix& lhs,
                                                     const VariableMatrix& rhs);

  /**
   * Compound subtraction-assignment operator.
   *
   * @param rhs Variable to subtract.
   */
  VariableMatrix& operator-=(const VariableMatrix& rhs);

  /**
   * Unary minus operator.
   *
   * @param lhs Operand for unary minus.
   */
  friend SLEIPNIR_DLLEXPORT VariableMatrix operator-(const VariableMatrix& lhs);

  /**
   * Implicit conversion operator from 1x1 VariableMatrix to Variable.
   */
  operator Variable() const;  // NOLINT

  /**
   * Returns the transpose of the variable matrix.
   */
  VariableMatrix T() const;

  /**
   * Returns number of rows in the matrix.
   */
  int Rows() const;

  /**
   * Returns number of columns in the matrix.
   */
  int Cols() const;

  /**
   * Returns an element of the variable matrix.
   *
   * @param row The row of the element to return.
   * @param col The column of the element to return.
   */
  double Value(int row, int col) const;

  /**
   * Returns a row of the variable column vector.
   *
   * @param index The index of the element to return.
   */
  double Value(int index) const;

  /**
   * Returns the contents of the variable matrix.
   */
  Eigen::MatrixXd Value() const;

  /**
   * Transforms the matrix coefficient-wise with an unary operator.
   *
   * @param unaryOp The unary operator to use for the transform operation.
   */
  template <std::invocable<const Variable&> UnaryOp>
  VariableMatrix CwiseTransform(UnaryOp&& unaryOp) const {
    VariableMatrix result{Rows(), Cols()};

    for (int row = 0; row < Rows(); ++row) {
      for (int col = 0; col < Cols(); ++col) {
        result(row, col) = unaryOp((*this)(row, col));
      }
    }

    return result;
  }

  class iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = Variable;
    using difference_type = std::ptrdiff_t;
    using pointer = Variable*;
    using reference = Variable&;

    iterator(VariableMatrix* mat, int row, int col)
        : m_mat{mat}, m_row{row}, m_col{col} {}

    iterator& operator++() {
      ++m_col;
      if (m_col == m_mat->Cols()) {
        m_col = 0;
        ++m_row;
      }
      return *this;
    }
    iterator operator++(int) {
      iterator retval = *this;
      ++(*this);
      return retval;
    }
    bool operator==(const iterator&) const = default;
    reference operator*() { return (*m_mat)(m_row, m_col); }

   private:
    VariableMatrix* m_mat;
    int m_row;
    int m_col;
  };

  class const_iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = Variable;
    using difference_type = std::ptrdiff_t;
    using pointer = Variable*;
    using const_reference = const Variable&;

    const_iterator(const VariableMatrix* mat, int row, int col)
        : m_mat{mat}, m_row{row}, m_col{col} {}

    const_iterator& operator++() {
      ++m_col;
      if (m_col == m_mat->Cols()) {
        m_col = 0;
        ++m_row;
      }
      return *this;
    }
    const_iterator operator++(int) {
      const_iterator retval = *this;
      ++(*this);
      return retval;
    }
    bool operator==(const const_iterator&) const = default;
    const_reference operator*() const { return (*m_mat)(m_row, m_col); }

   private:
    const VariableMatrix* m_mat;
    int m_row;
    int m_col;
  };

  /**
   * Returns begin iterator.
   */
  iterator begin() { return iterator(this, 0, 0); }

  /**
   * Returns end iterator.
   */
  iterator end() { return iterator(this, Rows(), 0); }

  /**
   * Returns begin iterator.
   */
  const_iterator begin() const { return const_iterator(this, 0, 0); }

  /**
   * Returns end iterator.
   */
  const_iterator end() const { return const_iterator(this, Rows(), 0); }

  /**
   * Returns begin iterator.
   */
  const_iterator cbegin() const { return const_iterator(this, 0, 0); }

  /**
   * Returns end iterator.
   */
  const_iterator cend() const { return const_iterator(this, Rows(), 0); }

  /**
   * Returns number of elements in matrix.
   */
  size_t size() const { return m_rows * m_cols; }

  /**
   * Returns a variable matrix filled with zeroes.
   *
   * @param rows The number of matrix rows.
   * @param cols The number of matrix columns.
   */
  static VariableMatrix Zero(int rows, int cols);

  /**
   * Returns a variable matrix filled with ones.
   *
   * @param rows The number of matrix rows.
   * @param cols The number of matrix columns.
   */
  static VariableMatrix Ones(int rows, int cols);

 private:
  std::vector<Variable> m_storage;
  int m_rows = 0;
  int m_cols = 0;
};

/**
 * Applies a coefficient-wise reduce operation to two matrices.
 *
 * @param lhs The left-hand side of the binary operator.
 * @param rhs The right-hand side of the binary operator.
 * @param binaryOp The binary operator to use for the reduce operation.
 */
template <std::invocable<const Variable&, const Variable&> BinaryOp>
VariableMatrix CwiseReduce(const VariableMatrix& lhs, const VariableMatrix& rhs,
                           BinaryOp&& binaryOp) {
  assert(lhs.Rows() == rhs.Rows());
  assert(lhs.Rows() == rhs.Rows());

  VariableMatrix result{lhs.Rows(), lhs.Cols()};

  for (int row = 0; row < lhs.Rows(); ++row) {
    for (int col = 0; col < lhs.Cols(); ++col) {
      result(row, col) = binaryOp(lhs(row, col), rhs(row, col));
    }
  }

  return result;
}

/**
 * Assemble a VariableMatrix from a nested list of blocks.
 *
 * Each row's blocks must have the same height, and the assembled block rows
 * must have the same width. For example, for the block matrix [[A, B], [C]] to
 * be constructible, the number of rows in A and B must match, and the number of
 * columns in [A, B] and [C] must match.
 *
 * @param list The nested list of blocks.
 */
SLEIPNIR_DLLEXPORT VariableMatrix
Block(std::initializer_list<std::initializer_list<VariableMatrix>> list);

/**
 * Assemble a VariableMatrix from a nested list of blocks.
 *
 * Each row's blocks must have the same height, and the assembled block rows
 * must have the same width. For example, for the block matrix [[A, B], [C]] to
 * be constructible, the number of rows in A and B must match, and the number of
 * columns in [A, B] and [C] must match.
 *
 * This overload is for Python bindings only.
 *
 * @param list The nested list of blocks.
 */
SLEIPNIR_DLLEXPORT VariableMatrix
Block(std::vector<std::vector<VariableMatrix>> list);

}  // namespace sleipnir
