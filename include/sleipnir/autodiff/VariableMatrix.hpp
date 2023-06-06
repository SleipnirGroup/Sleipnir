// Copyright (c) Sleipnir contributors

#pragma once

#include <cassert>
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
   * Constructs a VariableMatrix from an Eigen matrix.
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
          sum += Constant(lhs(i, k)) * rhs(k, j);
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
          sum += lhs(i, k) * Constant(rhs(k, j));
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
    VariableMatrix result{static_cast<int>(lhs.rows()),
                          static_cast<int>(lhs.cols())};

    for (int row = 0; row < result.Rows(); ++row) {
      for (int col = 0; col < result.Cols(); ++col) {
        result(row, col) = Constant(lhs(row, col)) + rhs(row, col);
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
    VariableMatrix result{lhs.Rows(), lhs.Cols()};

    for (int row = 0; row < result.Rows(); ++row) {
      for (int col = 0; col < result.Cols(); ++col) {
        result(row, col) = lhs(row, col) + Constant(rhs(row, col));
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
    VariableMatrix result{static_cast<int>(lhs.rows()),
                          static_cast<int>(lhs.cols())};

    for (int row = 0; row < result.Rows(); ++row) {
      for (int col = 0; col < result.Cols(); ++col) {
        result(row, col) = Constant(lhs(row, col)) - rhs(row, col);
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
    VariableMatrix result{lhs.Rows(), lhs.Cols()};

    for (int row = 0; row < result.Rows(); ++row) {
      for (int col = 0; col < result.Cols(); ++col) {
        result(row, col) = lhs(row, col) - Constant(rhs(row, col));
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

 private:
  std::vector<Variable> m_storage;
  int m_rows = 0;
  int m_cols = 0;
};

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
