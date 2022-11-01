// Copyright (c) Joshua Nichols and Tyler Veness

#pragma once

#include <cassert>
#include <utility>

#include <Eigen/Core>

#include "sleipnir/SymbolExports.h"
#include "sleipnir/autodiff/Variable.h"

namespace sleipnir {

class SLEIPNIR_DLLEXPORT Mat;

template <typename Mat>
class VariableBlock {
 public:
  VariableBlock(Mat& mat);  // NOLINT

  VariableBlock(Mat& mat, int rowOffset, int colOffset, int blockRows,
                int blockCols);

  VariableBlock<Mat>& operator=(double value);

  template <int _Rows, int _Cols>
  VariableBlock<Mat>& operator=(
      const Eigen::Matrix<double, _Rows, _Cols>& values) {
    for (size_t row = 0; row < _Rows; ++row) {
      for (size_t col = 0; col < _Cols; ++col) {
        Autodiff(row, col) = values(row, col);
      }
    }

    return *this;
  }

  template <int _Rows, int _Cols>
  VariableBlock<Mat>& operator=(Eigen::Matrix<double, _Rows, _Cols>&& values) {
    for (size_t row = 0; row < _Rows; ++row) {
      for (size_t col = 0; col < _Cols; ++col) {
        Autodiff(row, col) = values(row, col);
      }
    }

    return *this;
  }

  template <int _Rows, int _Cols>
  VariableBlock<Mat>& operator=(
      const Eigen::Matrix<autodiff::Variable, _Rows, _Cols>& values) {
    for (int row = 0; row < m_blockRows; ++row) {
      for (int col = 0; col < m_blockCols; ++col) {
        Autodiff(row, col) = values(row, col);
      }
    }
    return *this;
  }

  template <int _Rows, int _Cols>
  VariableBlock<Mat>& operator=(
      Eigen::Matrix<autodiff::Variable, _Rows, _Cols>&& values) {
    for (int row = 0; row < m_blockRows; ++row) {
      for (int col = 0; col < m_blockCols; ++col) {
        Autodiff(row, col) = std::move(values(row, col));
      }
    }
    return *this;
  }

  VariableBlock<Mat>& operator=(const Mat& values) {
    for (int row = 0; row < m_blockRows; ++row) {
      for (int col = 0; col < m_blockCols; ++col) {
        Autodiff(row, col) = values.Autodiff(row, col);
      }
    }
    return *this;
  }

  VariableBlock<Mat>& operator=(Mat&& values) {
    for (int row = 0; row < m_blockRows; ++row) {
      for (int col = 0; col < m_blockCols; ++col) {
        Autodiff(row, col) = std::move(values.Autodiff(row, col));
      }
    }
    return *this;
  }

  VariableBlock<Mat> operator()(int row, int col);

  VariableBlock<const Mat> operator()(int row, int col) const;

  VariableBlock<Mat> operator()(int row);

  VariableBlock<const Mat> operator()(int row) const;

  /**
   * Returns a block slice of the variable matrix.
   *
   * @param rowOffset The row offset of the block selection.
   * @param colOffset The column offset of the block selection.
   * @param blockRows The number of rows in the block selection.
   * @param blockCols The number of columns in the block selection.
   */
  VariableBlock<Mat> Block(int rowOffset, int colOffset, int blockRows,
                           int blockCols);

  /**
   * Returns a block slice of the variable matrix.
   *
   * @param rowOffset The row offset of the block selection.
   * @param colOffset The column offset of the block selection.
   * @param blockRows The number of rows in the block selection.
   * @param blockCols The number of columns in the block selection.
   */
  const VariableBlock<const Mat> Block(int rowOffset, int colOffset,
                                       int blockRows, int blockCols) const;

  /**
   * Returns a row slice of the variable matrix.
   *
   * @param row The row to slice.
   */
  VariableBlock<Mat> Row(int row);

  /**
   * Returns a row slice of the variable matrix.
   *
   * @param row The row to slice.
   */
  VariableBlock<const Mat> Row(int row) const;

  /**
   * Returns a column slice of the variable matrix.
   *
   * @param col The column to slice.
   */
  VariableBlock<Mat> Col(int col);

  /**
   * Returns a column slice of the variable matrix.
   *
   * @param col The column to slice.
   */
  VariableBlock<const Mat> Col(int col) const;

  /**
   * Compound matrix multiplication-assignment operator.
   *
   * @param rhs Variable to multiply.
   */
  VariableBlock<Mat>& operator*=(const VariableBlock<Mat>& rhs);

  /**
   * Compound matrix multiplication-assignment operator (only enabled when lhs
   * is a scalar).
   *
   * @param rhs Variable to multiply.
   */
  VariableBlock& operator*=(double rhs);

  /**
   * Compound matrix division-assignment operator (only enabled when rhs
   * is a scalar).
   *
   * @param rhs Variable to divide.
   */
  VariableBlock<Mat>& operator/=(const VariableBlock<Mat>& rhs);

  /**
   * Compound matrix division-assignment operator (only enabled when rhs
   * is a scalar).
   *
   * @param rhs Variable to divide.
   */
  VariableBlock<Mat>& operator/=(double rhs);

  /**
   * Compound addition-assignment operator.
   *
   * @param rhs Variable to add.
   */
  VariableBlock<Mat>& operator+=(const VariableBlock<Mat>& rhs);

  /**
   * Compound subtraction-assignment operator.
   *
   * @param rhs Variable to subtract.
   */
  VariableBlock<Mat>& operator-=(const VariableBlock<Mat>& rhs);

  /**
   * Returns the transpose of the variable matrix.
   */
  Mat Transpose() const;

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
   * Returns the autodiff variable backing a matrix entry.
   */
  autodiff::Variable& Autodiff(int row, int col);

  /**
   * Returns the autodiff variable backing a matrix entry.
   */
  const autodiff::Variable& Autodiff(int row, int col) const;

 private:
  Mat* m_mat = nullptr;
  int m_rowOffset = 0;
  int m_colOffset = 0;
  int m_blockRows = 0;
  int m_blockCols = 0;
};

/**
 * std::abs() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat abs(const VariableBlock<Mat>& x);

/**
 * std::acos() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat acos(const VariableBlock<Mat>& x);

/**
 * std::asin() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat asin(const VariableBlock<Mat>& x);

/**
 * std::atan() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat atan(const VariableBlock<Mat>& x);

/**
 * std::atan2() for VariableMatrices.
 *
 * The function is applied element-wise to the arguments.
 *
 * @param y The y argument.
 * @param x The x argument.
 */
template <typename Mat>
Mat atan2(const VariableBlock<Mat>& y, const VariableBlock<Mat>& x);

/**
 * std::cos() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat cos(const VariableBlock<Mat>& x);

/**
 * std::cosh() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat cosh(const VariableBlock<Mat>& x);

/**
 * std::erf() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat erf(const VariableBlock<Mat>& x);

/**
 * std::exp() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat exp(const VariableBlock<Mat>& x);

/**
 * std::hypot() for VariableMatrices.
 *
 * The function is applied element-wise to the arguments.
 *
 * @param x The x argument.
 * @param y The y argument.
 */
template <typename Mat>
Mat hypot(const VariableBlock<Mat>& x, const VariableBlock<Mat>& y);

/**
 * std::log() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat log(const VariableBlock<Mat>& x);

/**
 * std::log10() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat log10(const VariableBlock<Mat>& x);

/**
 * std::pow() for VariableMatrices.
 *
 * The function is applied element-wise to the arguments.
 *
 * @param base The base.
 * @param power The power.
 */
template <typename Mat>
Mat pow(const VariableBlock<Mat>& base, const VariableBlock<Mat>& power);

/**
 * std::sin() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat sin(const VariableBlock<Mat>& x);

/**
 * std::sinh() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat sinh(const VariableBlock<Mat>& x);

/**
 * std::sqrt() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat sqrt(const VariableBlock<Mat>& x);

/**
 * std::tan() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat tan(const VariableBlock<Mat>& x);

/**
 * std::tanh() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat tanh(const VariableBlock<Mat>& x);

}  // namespace sleipnir

#include "sleipnir/optimization/VariableBlock.inc"
