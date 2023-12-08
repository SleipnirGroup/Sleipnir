// Copyright (c) Sleipnir contributors

#pragma once

#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>

#include <Eigen/Core>

#include "sleipnir/SymbolExports.hpp"
#include "sleipnir/autodiff/Variable.hpp"

namespace sleipnir {

/**
 * A submatrix of autodiff variables with reference semantics.
 *
 * @tparam Mat The type of the matrix whose storage this class points to.
 */
template <typename Mat>
class VariableBlock {
 public:
  /**
   * Constructs a Variable block pointing to all of the given matrix.
   *
   * @param mat The matrix to which to point.
   */
  VariableBlock(Mat& mat)  // NOLINT
      : m_mat{&mat}, m_blockRows{mat.Rows()}, m_blockCols{mat.Cols()} {}

  /**
   * Constructs a Variable block pointing to a subset of the given matrix.
   *
   * @param mat The matrix to which to point.
   * @param rowOffset The block's row offset.
   * @param colOffset The block's column offset.
   * @param blockRows The number of rows in the block.
   * @param blockCols The number of columns in the block.
   */
  VariableBlock(Mat& mat, int rowOffset, int colOffset, int blockRows,
                int blockCols)
      : m_mat{&mat},
        m_rowOffset{rowOffset},
        m_colOffset{colOffset},
        m_blockRows{blockRows},
        m_blockCols{blockCols} {}

  /**
   * Assigns a double to the block.
   *
   * This only works for blocks with one row and one column.
   */
  VariableBlock<Mat>& operator=(double value) {
    assert(Rows() == 1 && Cols() == 1);

    (*this)(0, 0) = Constant(value);

    return *this;
  }

  /**
   * Assigns a double to the block.
   *
   * This only works for blocks with one row and one column.
   */
  VariableBlock<Mat>& SetValue(double value) {
    assert(Rows() == 1 && Cols() == 1);

    (*this)(0, 0).SetValue(value);

    return *this;
  }

  /**
   * Copy constructs a VariableBlock to the block.
   */
  VariableBlock(const VariableBlock<Mat>& values) {
    m_mat = values.m_mat;
    m_rowOffset = values.m_rowOffset;
    m_colOffset = values.m_colOffset;
    m_blockRows = values.m_blockRows;
    m_blockCols = values.m_blockCols;
  }

  /**
   * Assigns a VariableBlock to the block.
   */
  VariableBlock<Mat>& operator=(VariableBlock<Mat>& values) {
    if (this == &values) {
      return *this;
    }

    if (m_mat == nullptr) {
      m_mat = values.m_mat;
      m_rowOffset = values.m_rowOffset;
      m_colOffset = values.m_colOffset;
      m_blockRows = values.m_blockRows;
      m_blockCols = values.m_blockCols;
    } else {
      assert(Rows() == values.Rows());
      assert(Cols() == values.Cols());

      for (int row = 0; row < Rows(); ++row) {
        for (int col = 0; col < Cols(); ++col) {
          (*this)(row, col) = values(row, col);
        }
      }
    }

    return *this;
  }

  /**
   * Assigns an Eigen matrix to the block.
   */
  template <typename Derived>
  VariableBlock<Mat>& operator=(const Eigen::MatrixBase<Derived>& values) {
    assert(Rows() == values.rows());
    assert(Cols() == values.cols());

    for (int row = 0; row < Rows(); ++row) {
      for (int col = 0; col < Cols(); ++col) {
        if constexpr (std::same_as<typename Derived::Scalar, double>) {
          (*this)(row, col) = Variable{
              MakeExpression(values(row, col), ExpressionType::kConstant)};
        } else {
          (*this)(row, col) = values(row, col);
        }
      }
    }

    return *this;
  }

  /**
   * Sets block's internal values.
   */
  template <typename Derived>
    requires std::same_as<typename Derived::Scalar, double>
  VariableBlock<Mat>& SetValue(const Eigen::MatrixBase<Derived>& values) {
    assert(Rows() == values.rows());
    assert(Cols() == values.cols());

    for (int row = 0; row < Rows(); ++row) {
      for (int col = 0; col < Cols(); ++col) {
        (*this)(row, col).SetValue(values(row, col));
      }
    }

    return *this;
  }

  /**
   * Assigns a VariableMatrix to the block.
   */
  VariableBlock<Mat>& operator=(const Mat& values) {
    assert(Rows() == values.Rows());
    assert(Cols() == values.Cols());

    for (int row = 0; row < Rows(); ++row) {
      for (int col = 0; col < Cols(); ++col) {
        (*this)(row, col) = values(row, col);
      }
    }
    return *this;
  }

  /**
   * Assigns a VariableMatrix to the block.
   */
  VariableBlock<Mat>& operator=(Mat&& values) {
    assert(Rows() == values.Rows());
    assert(Cols() == values.Cols());

    for (int row = 0; row < Rows(); ++row) {
      for (int col = 0; col < Cols(); ++col) {
        (*this)(row, col) = std::move(values(row, col));
      }
    }
    return *this;
  }

  /**
   * Returns a scalar subblock at the given row and column.
   *
   * @param row The scalar subblock's row.
   * @param col The scalar subblock's column.
   */
  template <typename Mat2 = Mat>
    requires(!std::is_const_v<Mat2>)
  Variable& operator()(int row, int col) {
    assert(row >= 0 && row < Rows());
    assert(col >= 0 && col < Cols());
    return (*m_mat)(m_rowOffset + row, m_colOffset + col);
  }

  /**
   * Returns a scalar subblock at the given row and column.
   *
   * @param row The scalar subblock's row.
   * @param col The scalar subblock's column.
   */
  const Variable& operator()(int row, int col) const {
    assert(row >= 0 && row < Rows());
    assert(col >= 0 && col < Cols());
    return (*m_mat)(m_rowOffset + row, m_colOffset + col);
  }

  /**
   * Returns a scalar subblock at the given row.
   *
   * @param row The scalar subblock's row.
   */
  template <typename Mat2 = Mat>
    requires(!std::is_const_v<Mat2>)
  Variable& operator()(int row) {
    assert(row >= 0 && row < Rows() * Cols());
    return (*m_mat)(row);
  }

  /**
   * Returns a scalar subblock at the given row.
   *
   * @param row The scalar subblock's row.
   */
  const Variable& operator()(int row) const {
    assert(row >= 0 && row < Rows() * Cols());
    return (*m_mat)(row);
  }

  /**
   * Returns a block slice of the variable matrix.
   *
   * @param rowOffset The row offset of the block selection.
   * @param colOffset The column offset of the block selection.
   * @param blockRows The number of rows in the block selection.
   * @param blockCols The number of columns in the block selection.
   */
  VariableBlock<Mat> Block(int rowOffset, int colOffset, int blockRows,
                           int blockCols) {
    assert(rowOffset >= 0 && rowOffset <= Rows());
    assert(colOffset >= 0 && colOffset <= Cols());
    assert(blockRows >= 0 && blockRows <= Rows() - rowOffset);
    assert(blockCols >= 0 && blockCols <= Cols() - colOffset);
    return VariableBlock{*m_mat, rowOffset, colOffset, blockRows, blockCols};
  }

  /**
   * Returns a block slice of the variable matrix.
   *
   * @param rowOffset The row offset of the block selection.
   * @param colOffset The column offset of the block selection.
   * @param blockRows The number of rows in the block selection.
   * @param blockCols The number of columns in the block selection.
   */
  const VariableBlock<const Mat> Block(int rowOffset, int colOffset,
                                       int blockRows, int blockCols) const {
    assert(rowOffset >= 0 && rowOffset <= Rows());
    assert(colOffset >= 0 && colOffset <= Cols());
    assert(blockRows >= 0 && blockRows <= Rows() - rowOffset);
    assert(blockCols >= 0 && blockCols <= Cols() - colOffset);
    return VariableBlock{*m_mat, rowOffset, colOffset, blockRows, blockCols};
  }

  /**
   * Returns a row slice of the variable matrix.
   *
   * @param row The row to slice.
   */
  VariableBlock<Mat> Row(int row) {
    assert(row >= 0 && row < Rows());
    return Block(row, 0, 1, Cols());
  }

  /**
   * Returns a row slice of the variable matrix.
   *
   * @param row The row to slice.
   */
  VariableBlock<const Mat> Row(int row) const {
    assert(row >= 0 && row < Rows());
    return Block(row, 0, 1, Cols());
  }

  /**
   * Returns a column slice of the variable matrix.
   *
   * @param col The column to slice.
   */
  VariableBlock<Mat> Col(int col) {
    assert(col >= 0 && col < Cols());
    return Block(0, col, Rows(), 1);
  }

  /**
   * Returns a column slice of the variable matrix.
   *
   * @param col The column to slice.
   */
  VariableBlock<const Mat> Col(int col) const {
    assert(col >= 0 && col < Cols());
    return Block(0, col, Rows(), 1);
  }

  /**
   * Compound matrix multiplication-assignment operator.
   *
   * @param rhs Variable to multiply.
   */
  VariableBlock<Mat>& operator*=(const VariableBlock<Mat>& rhs) {
    assert(Cols() == rhs.Rows() && Cols() == rhs.Cols());

    for (int i = 0; i < Rows(); ++i) {
      for (int j = 0; j < rhs.Cols(); ++j) {
        Variable sum;
        for (int k = 0; k < Cols(); ++k) {
          sum += (*this)(i, k) * rhs(k, j);
        }
        (*this)(i, j) = sum;
      }
    }

    return *this;
  }

  /**
   * Compound matrix multiplication-assignment operator (only enabled when lhs
   * is a scalar).
   *
   * @param rhs Variable to multiply.
   */
  VariableBlock& operator*=(double rhs) {
    for (int row = 0; row < Rows(); ++row) {
      for (int col = 0; col < Cols(); ++col) {
        (*this)(row, col) *= Constant(rhs);
      }
    }

    return *this;
  }

  /**
   * Compound matrix division-assignment operator (only enabled when rhs
   * is a scalar).
   *
   * @param rhs Variable to divide.
   */
  VariableBlock<Mat>& operator/=(const VariableBlock<Mat>& rhs) {
    assert(rhs.Rows() == 1 && rhs.Cols() == 1);

    for (int row = 0; row < Rows(); ++row) {
      for (int col = 0; col < Cols(); ++col) {
        (*this)(row, col) /= rhs(0, 0);
      }
    }

    return *this;
  }

  /**
   * Compound matrix division-assignment operator (only enabled when rhs
   * is a scalar).
   *
   * @param rhs Variable to divide.
   */
  VariableBlock<Mat>& operator/=(double rhs) {
    for (int row = 0; row < Rows(); ++row) {
      for (int col = 0; col < Cols(); ++col) {
        (*this)(row, col) /= Constant(rhs);
      }
    }

    return *this;
  }

  /**
   * Compound addition-assignment operator.
   *
   * @param rhs Variable to add.
   */
  VariableBlock<Mat>& operator+=(const VariableBlock<Mat>& rhs) {
    for (int row = 0; row < Rows(); ++row) {
      for (int col = 0; col < Cols(); ++col) {
        (*this)(row, col) += rhs(row, col);
      }
    }

    return *this;
  }

  /**
   * Compound subtraction-assignment operator.
   *
   * @param rhs Variable to subtract.
   */
  VariableBlock<Mat>& operator-=(const VariableBlock<Mat>& rhs) {
    for (int row = 0; row < Rows(); ++row) {
      for (int col = 0; col < Cols(); ++col) {
        (*this)(row, col) -= rhs(row, col);
      }
    }

    return *this;
  }

  /**
   * Returns the transpose of the variable matrix.
   */
  Mat T() const {
    Mat result{Cols(), Rows()};

    for (int row = 0; row < Rows(); ++row) {
      for (int col = 0; col < Cols(); ++col) {
        result(col, row) = (*this)(row, col);
      }
    }

    return result;
  }

  /**
   * Returns number of rows in the matrix.
   */
  int Rows() const { return m_blockRows; }

  /**
   * Returns number of columns in the matrix.
   */
  int Cols() const { return m_blockCols; }

  /**
   * Returns an element of the variable matrix.
   *
   * @param row The row of the element to return.
   * @param col The column of the element to return.
   */
  double Value(int row, int col) const {
    assert(row >= 0 && row < Rows());
    assert(col >= 0 && col < Cols());
    return (*m_mat)(m_rowOffset + row, m_colOffset + col).Value();
  }

  /**
   * Returns a row of the variable column vector.
   *
   * @param index The index of the element to return.
   */
  double Value(int index) const {
    assert(index >= 0 && index < Rows() * Cols());
    return (*m_mat)(m_rowOffset + index / m_blockCols,
                    m_colOffset + index % m_blockCols)
        .Value();
  }

  /**
   * Returns the contents of the variable matrix.
   */
  Eigen::MatrixXd Value() const {
    Eigen::MatrixXd result{Rows(), Cols()};

    for (int row = 0; row < Rows(); ++row) {
      for (int col = 0; col < Cols(); ++col) {
        result(row, col) = Value(row, col);
      }
    }

    return result;
  }

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
Mat abs(const VariableBlock<Mat>& x) {
  std::remove_cv_t<Mat> result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::abs(x(row, col));
    }
  }

  return result;
}

/**
 * std::acos() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat acos(const VariableBlock<Mat>& x) {
  std::remove_cv_t<Mat> result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::acos(x(row, col));
    }
  }

  return result;
}

/**
 * std::asin() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat asin(const VariableBlock<Mat>& x) {
  std::remove_cv_t<Mat> result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::asin(x(row, col));
    }
  }

  return result;
}

/**
 * std::atan() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat atan(const VariableBlock<Mat>& x) {
  std::remove_cv_t<Mat> result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::atan(x(row, col));
    }
  }

  return result;
}

/**
 * std::atan2() for VariableMatrices.
 *
 * The function is applied element-wise to the arguments.
 *
 * @param y The y argument.
 * @param x The x argument.
 */
template <typename Mat>
Mat atan2(const VariableBlock<Mat>& y, const VariableBlock<Mat>& x) {
  std::remove_cv_t<Mat> result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::atan2(y(row, col), x(row, col));
    }
  }

  return result;
}

/**
 * std::cos() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat cos(const VariableBlock<Mat>& x) {
  std::remove_cv_t<Mat> result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::cos(x(row, col));
    }
  }

  return result;
}

/**
 * std::cosh() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat cosh(const VariableBlock<Mat>& x) {
  Mat result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::cosh(x(row, col));
    }
  }

  return result;
}

/**
 * std::erf() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat erf(const VariableBlock<Mat>& x) {
  std::remove_cv_t<Mat> result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::erf(x(row, col));
    }
  }

  return result;
}

/**
 * std::exp() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat exp(const VariableBlock<Mat>& x) {
  std::remove_cv_t<Mat> result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::exp(x(row, col));
    }
  }

  return result;
}

/**
 * std::hypot() for VariableMatrices.
 *
 * The function is applied element-wise to the arguments.
 *
 * @param x The x argument.
 * @param y The y argument.
 */
template <typename Mat>
Mat hypot(const VariableBlock<Mat>& x, const VariableBlock<Mat>& y) {
  std::remove_cv_t<Mat> result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::hypot(x(row, col), y(row, col));
    }
  }

  return result;
}

/**
 * std::log() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat log(const VariableBlock<Mat>& x) {
  std::remove_cv_t<Mat> result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::log(x(row, col));
    }
  }

  return result;
}

/**
 * std::log10() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat log10(const VariableBlock<Mat>& x) {
  std::remove_cv_t<Mat> result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::log10(x(row, col));
    }
  }

  return result;
}

/**
 * std::pow() for VariableMatrices.
 *
 * The function is applied element-wise to the arguments.
 *
 * @param base The base.
 * @param power The power.
 */
template <typename Mat>
Mat pow(const VariableBlock<Mat>& base, const VariableBlock<Mat>& power) {
  Mat result{base.Rows(), base.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::pow(base(row, col), power(row, col));
    }
  }

  return result;
}

/**
 * sign() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat sign(const VariableBlock<Mat>& x) {
  std::remove_cv_t<Mat> result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::sign(x(row, col));
    }
  }

  return result;
}

/**
 * std::sin() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat sin(const VariableBlock<Mat>& x) {
  std::remove_cv_t<Mat> result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::sin(x(row, col));
    }
  }

  return result;
}

/**
 * std::sinh() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat sinh(const VariableBlock<Mat>& x) {
  std::remove_cv_t<Mat> result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::sinh(x(row, col));
    }
  }

  return result;
}

/**
 * std::sqrt() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat sqrt(const VariableBlock<Mat>& x) {
  std::remove_cv_t<Mat> result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::sqrt(x(row, col));
    }
  }

  return result;
}

/**
 * std::tan() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat tan(const VariableBlock<Mat>& x) {
  std::remove_cv_t<Mat> result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::tan(x(row, col));
    }
  }

  return result;
}

/**
 * std::tanh() for VariableMatrices.
 *
 * The function is applied element-wise to the argument.
 *
 * @param x The argument.
 */
template <typename Mat>
Mat tanh(const VariableBlock<Mat>& x) {
  std::remove_cv_t<Mat> result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::tanh(x(row, col));
    }
  }

  return result;
}

}  // namespace sleipnir
