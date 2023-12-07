// Copyright (c) Sleipnir contributors

#include "sleipnir/autodiff/VariableMatrix.hpp"

#include <cassert>

#include "sleipnir/autodiff/Expression.hpp"

namespace sleipnir {

VariableMatrix::VariableMatrix(int rows, int cols)
    : m_rows{rows}, m_cols{cols} {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      m_storage.emplace_back();
    }
  }
}

VariableMatrix::VariableMatrix(double value) : m_rows{1}, m_cols{1} {
  m_storage.emplace_back(value);
}

VariableMatrix::VariableMatrix(int value) : m_rows{1}, m_cols{1} {
  m_storage.emplace_back(value);
}

VariableMatrix& VariableMatrix::operator=(double value) {
  assert(Rows() == 1 && Cols() == 1);

  (*this)(0, 0) = value;

  return *this;
}

VariableMatrix& VariableMatrix::operator=(int value) {
  assert(Rows() == 1 && Cols() == 1);

  (*this)(0, 0) = value;

  return *this;
}

VariableMatrix::VariableMatrix(const Variable& variable)
    : m_rows{1}, m_cols{1} {
  m_storage.emplace_back(variable);
}

VariableMatrix::VariableMatrix(Variable&& variable) : m_rows{1}, m_cols{1} {
  m_storage.emplace_back(std::move(variable));
}

VariableMatrix::VariableMatrix(const VariableBlock<VariableMatrix>& values)
    : m_rows{values.Rows()}, m_cols{values.Cols()} {
  for (int row = 0; row < Rows(); ++row) {
    for (int col = 0; col < Cols(); ++col) {
      m_storage.emplace_back(values(row, col));
    }
  }
}

VariableMatrix::VariableMatrix(
    const VariableBlock<const VariableMatrix>& values)
    : m_rows{values.Rows()}, m_cols{values.Cols()} {
  for (int row = 0; row < Rows(); ++row) {
    for (int col = 0; col < Cols(); ++col) {
      m_storage.emplace_back(values(row, col));
    }
  }
}

Variable& VariableMatrix::operator()(int row, int col) {
  assert(row >= 0 && row < Rows());
  assert(col >= 0 && col < Cols());
  return m_storage[row * Cols() + col];
}

const Variable& VariableMatrix::operator()(int row, int col) const {
  assert(row >= 0 && row < Rows());
  assert(col >= 0 && col < Cols());
  return m_storage[row * Cols() + col];
}

Variable& VariableMatrix::operator()(int row) {
  assert(row >= 0 && row < Rows() * Cols());
  return m_storage[row * Cols()];
}

const Variable& VariableMatrix::operator()(int row) const {
  assert(row >= 0 && row < Rows() * Cols());
  return m_storage[row * Cols()];
}

VariableBlock<VariableMatrix> VariableMatrix::Block(int rowOffset,
                                                    int colOffset,
                                                    int blockRows,
                                                    int blockCols) {
  assert(rowOffset >= 0 && rowOffset <= Rows());
  assert(colOffset >= 0 && colOffset <= Cols());
  assert(blockRows >= 0 && blockRows <= Rows() - rowOffset);
  assert(blockCols >= 0 && blockCols <= Cols() - colOffset);
  return VariableBlock{*this, rowOffset, colOffset, blockRows, blockCols};
}

const VariableBlock<const VariableMatrix> VariableMatrix::Block(
    int rowOffset, int colOffset, int blockRows, int blockCols) const {
  assert(rowOffset >= 0 && rowOffset <= Rows());
  assert(colOffset >= 0 && colOffset <= Cols());
  assert(blockRows >= 0 && blockRows <= Rows() - rowOffset);
  assert(blockCols >= 0 && blockCols <= Cols() - colOffset);
  return VariableBlock{*this, rowOffset, colOffset, blockRows, blockCols};
}

VariableBlock<VariableMatrix> VariableMatrix::Segment(int offset, int length) {
  assert(offset >= 0 && offset < Rows() * Cols());
  assert(length >= 0 && length <= Rows() * Cols() - offset);
  return Block(offset, 0, length, 1);
}

const VariableBlock<const VariableMatrix> VariableMatrix::Segment(
    int offset, int length) const {
  assert(offset >= 0 && offset < Rows() * Cols());
  assert(length >= 0 && length <= Rows() * Cols() - offset);
  return Block(offset, 0, length, 1);
}

VariableBlock<VariableMatrix> VariableMatrix::Row(int row) {
  assert(row >= 0 && row < Rows());
  return Block(row, 0, 1, Cols());
}

const VariableBlock<const VariableMatrix> VariableMatrix::Row(int row) const {
  assert(row >= 0 && row < Rows());
  return Block(row, 0, 1, Cols());
}

VariableBlock<VariableMatrix> VariableMatrix::Col(int col) {
  assert(col >= 0 && col < Cols());
  return Block(0, col, Rows(), 1);
}

const VariableBlock<const VariableMatrix> VariableMatrix::Col(int col) const {
  assert(col >= 0 && col < Cols());
  return Block(0, col, Rows(), 1);
}

VariableMatrix operator*(const VariableMatrix& lhs, const VariableMatrix& rhs) {
  assert(lhs.Cols() == rhs.Rows());

  VariableMatrix result{lhs.Rows(), rhs.Cols()};

  for (int i = 0; i < lhs.Rows(); ++i) {
    for (int j = 0; j < rhs.Cols(); ++j) {
      Variable sum;
      for (int k = 0; k < lhs.Cols(); ++k) {
        sum += lhs(i, k) * rhs(k, j);
      }
      result(i, j) = sum;
    }
  }

  return result;
}

VariableMatrix operator*(const VariableMatrix& lhs, const Variable& rhs) {
  VariableMatrix result{lhs.Rows(), lhs.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = lhs(row, col) * rhs;
    }
  }

  return result;
}

VariableMatrix operator*(const VariableMatrix& lhs, double rhs) {
  return lhs * Constant(rhs);
}

VariableMatrix operator*(const Variable& lhs, const VariableMatrix& rhs) {
  VariableMatrix result{rhs.Rows(), rhs.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = rhs(row, col) * lhs;
    }
  }

  return result;
}

VariableMatrix operator*(double lhs, const VariableMatrix& rhs) {
  return Constant(lhs) * rhs;
}

VariableMatrix& VariableMatrix::operator*=(const VariableMatrix& rhs) {
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

VariableMatrix& VariableMatrix::operator*=(double rhs) {
  for (int row = 0; row < Rows(); ++row) {
    for (int col = 0; col < Cols(); ++col) {
      (*this)(row, col) *= Constant(rhs);
    }
  }

  return *this;
}

VariableMatrix operator/(const VariableMatrix& lhs, const VariableMatrix& rhs) {
  assert(rhs.Rows() == 1 && rhs.Cols() == 1);

  VariableMatrix result{lhs.Rows(), lhs.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = lhs(row, col) / rhs(0, 0);
    }
  }

  return result;
}

VariableMatrix operator/(const VariableMatrix& lhs, double rhs) {
  VariableMatrix result{lhs.Rows(), lhs.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = lhs(row, col) / Constant(rhs);
    }
  }

  return result;
}

VariableMatrix& VariableMatrix::operator/=(const VariableMatrix& rhs) {
  for (int row = 0; row < Rows(); ++row) {
    for (int col = 0; col < Cols(); ++col) {
      (*this)(row, col) /= rhs(0, 0);
    }
  }

  return *this;
}

VariableMatrix& VariableMatrix::operator/=(double rhs) {
  for (int row = 0; row < Rows(); ++row) {
    for (int col = 0; col < Cols(); ++col) {
      (*this)(row, col) /= Constant(rhs);
    }
  }

  return *this;
}

VariableMatrix operator+(const VariableMatrix& lhs, const VariableMatrix& rhs) {
  VariableMatrix result{lhs.Rows(), lhs.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = lhs(row, col) + rhs(row, col);
    }
  }

  return result;
}

VariableMatrix& VariableMatrix::operator+=(const VariableMatrix& rhs) {
  for (int row = 0; row < Rows(); ++row) {
    for (int col = 0; col < Cols(); ++col) {
      (*this)(row, col) += rhs(row, col);
    }
  }

  return *this;
}

VariableMatrix operator-(const VariableMatrix& lhs, const VariableMatrix& rhs) {
  VariableMatrix result{lhs.Rows(), lhs.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = lhs(row, col) - rhs(row, col);
    }
  }

  return result;
}

VariableMatrix& VariableMatrix::operator-=(const VariableMatrix& rhs) {
  for (int row = 0; row < Rows(); ++row) {
    for (int col = 0; col < Cols(); ++col) {
      (*this)(row, col) -= rhs(row, col);
    }
  }

  return *this;
}

VariableMatrix operator-(const VariableMatrix& lhs) {
  VariableMatrix result{lhs.Rows(), lhs.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = -lhs(row, col);
    }
  }

  return result;
}

VariableMatrix VariableMatrix::T() const {
  VariableMatrix result{Cols(), Rows()};

  for (int row = 0; row < Rows(); ++row) {
    for (int col = 0; col < Cols(); ++col) {
      result(col, row) = (*this)(row, col);
    }
  }

  return result;
}

VariableMatrix::operator Variable() const {
  assert(Rows() == 1 && Cols() == 1);
  return (*this)(0, 0);
}

int VariableMatrix::Rows() const {
  return m_rows;
}

int VariableMatrix::Cols() const {
  return m_cols;
}

double VariableMatrix::Value(int row, int col) const {
  assert(row >= 0 && row < Rows());
  assert(col >= 0 && col < Cols());
  return m_storage[row * Cols() + col].Value();
}

double VariableMatrix::Value(int index) const {
  assert(index >= 0 && index < Rows() * Cols());
  return m_storage[index].Value();
}

Eigen::MatrixXd VariableMatrix::Value() const {
  Eigen::MatrixXd result{Rows(), Cols()};

  for (int row = 0; row < Rows(); ++row) {
    for (int col = 0; col < Cols(); ++col) {
      result(row, col) = Value(row, col);
    }
  }

  return result;
}

VariableMatrix Block(
    std::initializer_list<std::initializer_list<VariableMatrix>> list) {
  // Get row and column counts for destination matrix
  int rows = 0;
  int cols = -1;
  for (const auto& row : list) {
    if (row.size() > 0) {
      rows += row.begin()->Rows();
    }

    // Get number of columns in this row
    int latestCols = 0;
    for (const auto& elem : row) {
      // Assert the first and latest row have the same height
      assert(row.begin()->Rows() == elem.Rows());

      latestCols += elem.Cols();
    }

    // If this is the first row, record the column count. Otherwise, assert the
    // first and latest column counts are the same.
    if (cols == -1) {
      cols = latestCols;
    } else {
      assert(cols == latestCols);
    }
  }

  VariableMatrix result{static_cast<int>(rows), static_cast<int>(cols)};

  int rowOffset = 0;
  for (const auto& row : list) {
    int colOffset = 0;
    for (const VariableMatrix& elem : row) {
      result.Block(rowOffset, colOffset, elem.Rows(), elem.Cols()) = elem;
      colOffset += elem.Cols();
    }
    rowOffset += row.begin()->Rows();
  }

  return result;
}

VariableMatrix abs(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::abs(x(row, col));
    }
  }

  return result;
}

VariableMatrix acos(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::acos(x(row, col));
    }
  }

  return result;
}

VariableMatrix asin(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::asin(x(row, col));
    }
  }

  return result;
}

VariableMatrix atan(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::atan(x(row, col));
    }
  }

  return result;
}

VariableMatrix atan2(const VariableMatrix& y, const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::atan2(y(row, col), x(row, col));
    }
  }

  return result;
}

VariableMatrix cos(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::cos(x(row, col));
    }
  }

  return result;
}

VariableMatrix cosh(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::cosh(x(row, col));
    }
  }

  return result;
}

VariableMatrix erf(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::erf(x(row, col));
    }
  }

  return result;
}

VariableMatrix exp(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::exp(x(row, col));
    }
  }

  return result;
}

VariableMatrix hypot(const VariableMatrix& x, const VariableMatrix& y) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::hypot(x(row, col), y(row, col));
    }
  }

  return result;
}

VariableMatrix log(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::log(x(row, col));
    }
  }

  return result;
}

VariableMatrix log10(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::log10(x(row, col));
    }
  }

  return result;
}

VariableMatrix pow(double base, const VariableMatrix& power) {
  VariableMatrix result{power.Rows(), power.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::pow(Constant(base), power(row, col));
    }
  }

  return result;
}

VariableMatrix pow(const VariableMatrix& base, double power) {
  VariableMatrix result{base.Rows(), base.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::pow(base(row, col), Constant(power));
    }
  }

  return result;
}

VariableMatrix pow(const VariableMatrix& base, const VariableMatrix& power) {
  VariableMatrix result{base.Rows(), base.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::pow(base(row, col), power(row, col));
    }
  }

  return result;
}

VariableMatrix sign(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::sign(x(row, col));
    }
  }

  return result;
}

VariableMatrix sin(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::sin(x(row, col));
    }
  }

  return result;
}

VariableMatrix sinh(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::sinh(x(row, col));
    }
  }

  return result;
}

VariableMatrix sqrt(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::sqrt(x(row, col));
    }
  }

  return result;
}

VariableMatrix tan(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::tan(x(row, col));
    }
  }

  return result;
}

VariableMatrix tanh(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result(row, col) = sleipnir::tanh(x(row, col));
    }
  }

  return result;
}

}  // namespace sleipnir
