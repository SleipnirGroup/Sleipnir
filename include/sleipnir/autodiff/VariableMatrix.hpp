// Copyright (c) Sleipnir contributors

#pragma once

#include <cassert>
#include <concepts>
#include <initializer_list>
#include <iterator>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "sleipnir/SymbolExports.hpp"
#include "sleipnir/autodiff/Variable.hpp"
#include "sleipnir/autodiff/VariableBlock.hpp"

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
   * Constructs a VariableMatrix with the given dimensions.
   *
   * The internal storage is uninitialized until it's assigned.
   *
   * @param rows The number of matrix rows.
   * @param cols The number of matrix columns.
   */
  VariableMatrix(int rows, int cols);

  /**
   * Constructs a scalar VariableMatrix from a double.
   */
  VariableMatrix(double value);  // NOLINT

  /**
   * Constructs a scalar VariableMatrix from an int.
   */
  VariableMatrix(int value);  // NOLINT

  /**
   * Assigns a double to a scalar VariableMatrix.
   */
  VariableMatrix& operator=(double value);

  /**
   * Assigns an int to a scalar VariableMatrix.
   */
  VariableMatrix& operator=(int value);

  /**
   * Assigns a double to a scalar VariableMatrix.
   */
  VariableMatrix& SetValue(double value);

  /**
   * Assigns an int to a scalar VariableMatrix.
   */
  VariableMatrix& SetValue(int value);

  /**
   * Constructs a scalar VariableMatrix from a nested list of Variables.
   *
   * @param list The nested list of Variables.
   */
  VariableMatrix(
      std::initializer_list<std::initializer_list<Variable>> list);  // NOLINT

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
        if constexpr (std::same_as<typename Derived::Scalar, double>) {
          m_storage.emplace_back(
              MakeExpressionPtr(values(row, col), ExpressionType::kConstant));
        } else {
          m_storage.emplace_back(values(row, col));
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
        if constexpr (std::same_as<typename Derived::Scalar, double>) {
          (*this)(row, col) = Variable{
              MakeExpressionPtr(values(row, col), ExpressionType::kConstant)};
        } else {
          (*this)(row, col) = values(row, col);
        }
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
  template <typename Derived>
  friend VariableMatrix operator*(const Eigen::MatrixBase<Derived>& lhs,
                                  const VariableMatrix& rhs) {
    assert(lhs.cols() == rhs.Rows());

    VariableMatrix result{static_cast<int>(lhs.rows()), rhs.Cols()};

    for (int i = 0; i < lhs.rows(); ++i) {
      for (int j = 0; j < rhs.Cols(); ++j) {
        Variable sum = 0.0;
        for (int k = 0; k < lhs.cols(); ++k) {
          sum += lhs(i, k) * rhs(k, j);
        }
        result(i, j) = sum;
      }
    }

    return result;
  }

  /**
   * Matrix multiplication operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  template <typename Derived>
  friend VariableMatrix operator*(const VariableMatrix& lhs,
                                  const Eigen::MatrixBase<Derived>& rhs) {
    assert(lhs.Cols() == rhs.rows());

    VariableMatrix result{lhs.Rows(), static_cast<int>(rhs.cols())};

    for (int i = 0; i < lhs.Rows(); ++i) {
      for (int j = 0; j < rhs.cols(); ++j) {
        Variable sum = 0.0;
        for (int k = 0; k < lhs.Cols(); ++k) {
          sum += lhs(i, k) * rhs(k, j);
        }
        result(i, j) = sum;
      }
    }

    return result;
  }

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
   * Compound matrix multiplication-assignment operator (only enabled when lhs
   * is a scalar).
   *
   * @param rhs Variable to multiply.
   */
  VariableMatrix& operator*=(double rhs);

  /**
   * Binary division operator (only enabled when rhs is a scalar).
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT VariableMatrix operator/(const VariableMatrix& lhs,
                                                     const VariableMatrix& rhs);

  /**
   * Binary division operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT VariableMatrix operator/(const VariableMatrix& lhs,
                                                     double rhs);

  /**
   * Compound matrix division-assignment operator (only enabled when rhs
   * is a scalar).
   *
   * @param rhs Variable to divide.
   */
  VariableMatrix& operator/=(const VariableMatrix& rhs);

  /**
   * Compound matrix division-assignment operator (only enabled when rhs
   * is a scalar).
   *
   * @param rhs Variable to divide.
   */
  VariableMatrix& operator/=(double rhs);

  /**
   * Binary addition operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  template <typename Derived>
  friend VariableMatrix operator+(const Eigen::MatrixBase<Derived>& lhs,
                                  const VariableMatrix& rhs) {
    assert(lhs.rows() == rhs.Rows());
    assert(lhs.cols() == rhs.Cols());

    VariableMatrix result{static_cast<int>(lhs.rows()),
                          static_cast<int>(lhs.cols())};

    for (int row = 0; row < result.Rows(); ++row) {
      for (int col = 0; col < result.Cols(); ++col) {
        result(row, col) = lhs(row, col) + rhs(row, col);
      }
    }

    return result;
  }

  /**
   * Binary addition operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  template <typename Derived>
  friend VariableMatrix operator+(const VariableMatrix& lhs,
                                  const Eigen::MatrixBase<Derived>& rhs) {
    assert(lhs.Rows() == rhs.rows());
    assert(lhs.Cols() == rhs.cols());

    VariableMatrix result{lhs.Rows(), lhs.Cols()};

    for (int row = 0; row < result.Rows(); ++row) {
      for (int col = 0; col < result.Cols(); ++col) {
        result(row, col) = lhs(row, col) + rhs(row, col);
      }
    }

    return result;
  }

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
  template <typename Derived>
  friend VariableMatrix operator-(const Eigen::MatrixBase<Derived>& lhs,
                                  const VariableMatrix& rhs) {
    assert(lhs.rows() == rhs.Rows());
    assert(lhs.cols() == rhs.Cols());

    VariableMatrix result{static_cast<int>(lhs.rows()),
                          static_cast<int>(lhs.cols())};

    for (int row = 0; row < result.Rows(); ++row) {
      for (int col = 0; col < result.Cols(); ++col) {
        result(row, col) = lhs(row, col) - rhs(row, col);
      }
    }

    return result;
  }

  /**
   * Binary subtraction operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  friend SLEIPNIR_DLLEXPORT VariableMatrix operator-(const VariableMatrix& lhs,
                                                     const VariableMatrix& rhs);

  /**
   * Binary subtraction operator.
   *
   * @param lhs Operator left-hand side.
   * @param rhs Operator right-hand side.
   */
  template <typename Derived>
  friend VariableMatrix operator-(const VariableMatrix& lhs,
                                  const Eigen::MatrixBase<Derived>& rhs) {
    assert(lhs.Rows() == rhs.rows());
    assert(lhs.Cols() == rhs.cols());

    VariableMatrix result{lhs.Rows(), lhs.Cols()};

    for (int row = 0; row < result.Rows(); ++row) {
      for (int col = 0; col < result.Cols(); ++col) {
        result(row, col) = lhs(row, col) - rhs(row, col);
      }
    }

    return result;
  }

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
template <typename BinaryOp>
VariableMatrix CwiseReduce(const VariableMatrix& lhs, const VariableMatrix& rhs,
                           BinaryOp binaryOp) {
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
 * std::abs() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT VariableMatrix abs(const VariableMatrix& x);

/**
 * std::acos() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT VariableMatrix acos(const VariableMatrix& x);

/**
 * std::asin() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT VariableMatrix asin(const VariableMatrix& x);

/**
 * std::atan() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT VariableMatrix atan(const VariableMatrix& x);

/**
 * std::atan2() for VariableMatrices.
 *
 * The function is applied element-wise to the arguments.
 *
 * @param y The y argument.
 * @param x The x argument.
 */
SLEIPNIR_DLLEXPORT VariableMatrix atan2(const VariableMatrix& y,
                                        const VariableMatrix& x);

/**
 * std::cos() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT VariableMatrix cos(const VariableMatrix& x);

/**
 * std::cosh() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT VariableMatrix cosh(const VariableMatrix& x);

/**
 * std::erf() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT VariableMatrix erf(const VariableMatrix& x);

/**
 * std::exp() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT VariableMatrix exp(const VariableMatrix& x);

/**
 * std::hypot() for VariableMatrices.
 *
 * The function is applied element-wise to the arguments.
 *
 * @param x The x argument.
 * @param y The y argument.
 */
SLEIPNIR_DLLEXPORT VariableMatrix hypot(const VariableMatrix& x,
                                        const VariableMatrix& y);

/**
 * std::hypot() for VariableMatrices.
 *
 * The function is applied element-wise to the arguments.
 *
 * @param x The x argument.
 * @param y The y argument.
 * @param z The z argument.
 */
SLEIPNIR_DLLEXPORT VariableMatrix hypot(const VariableMatrix& x,
                                        const VariableMatrix& y,
                                        const VariableMatrix& z);

/**
 * std::log() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT VariableMatrix log(const VariableMatrix& x);

/**
 * std::log10() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT VariableMatrix log10(const VariableMatrix& x);

/**
 * std::pow() for VariableMatrices.
 *
 * The function is applied element-wise to the arguments.
 *
 * @param base The base.
 * @param power The power.
 */
SLEIPNIR_DLLEXPORT VariableMatrix pow(double base, const VariableMatrix& power);

/**
 * std::pow() for VariableMatrices.
 *
 * The function is applied element-wise to the arguments.
 *
 * @param base The base.
 * @param power The power.
 */
SLEIPNIR_DLLEXPORT VariableMatrix pow(const VariableMatrix& base, double power);

/**
 * std::pow() for VariableMatrices.
 *
 * The function is applied element-wise to the arguments.
 *
 * @param base The base.
 * @param power The power.
 */
SLEIPNIR_DLLEXPORT VariableMatrix pow(const VariableMatrix& base,
                                      const VariableMatrix& power);

/**
 * sign() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT VariableMatrix sign(const VariableMatrix& x);

/**
 * std::sin() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT VariableMatrix sin(const VariableMatrix& x);

/**
 * std::sinh() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT VariableMatrix sinh(const VariableMatrix& x);

/**
 * std::sqrt() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT VariableMatrix sqrt(const VariableMatrix& x);

/**
 * std::tan() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT VariableMatrix tan(const VariableMatrix& x);

/**
 * std::tanh() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
SLEIPNIR_DLLEXPORT VariableMatrix tanh(const VariableMatrix& x);

}  // namespace sleipnir
