// Copyright (c) Joshua Nichols and Tyler Veness

#include "sleipnir/optimization/VariableMatrix.hpp"

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

VariableMatrix& VariableMatrix::operator=(double value) {
  assert(Rows() == 1 && Cols() == 1);

  Autodiff(0, 0) = value;

  return *this;
}

VariableMatrix::VariableMatrix(const autodiff::Variable& variable)
    : m_rows{1}, m_cols{1} {
  m_storage.emplace_back(variable);
}

VariableMatrix::VariableMatrix(autodiff::Variable&& variable)
    : m_rows{1}, m_cols{1} {
  m_storage.emplace_back(std::move(variable));
}

VariableMatrix::VariableMatrix(const VariableBlock<VariableMatrix>& values)
    : m_rows{values.Rows()}, m_cols{values.Cols()} {
  for (int row = 0; row < Rows(); ++row) {
    for (int col = 0; col < Cols(); ++col) {
      m_storage.emplace_back(values.Autodiff(row, col));
    }
  }
}

VariableMatrix::VariableMatrix(
    const VariableBlock<const VariableMatrix>& values)
    : m_rows{values.Rows()}, m_cols{values.Cols()} {
  for (int row = 0; row < Rows(); ++row) {
    for (int col = 0; col < Cols(); ++col) {
      m_storage.emplace_back(values.Autodiff(row, col));
    }
  }
}

VariableBlock<VariableMatrix> VariableMatrix::operator()(int row, int col) {
  assert(row < Rows() && col < Cols());
  return Block(row, col, 1, 1);
}

VariableBlock<const VariableMatrix> VariableMatrix::operator()(int row,
                                                               int col) const {
  assert(row < Rows() && col < Cols());
  return Block(row, col, 1, 1);
}

VariableBlock<VariableMatrix> VariableMatrix::operator()(int row) {
  return Block(row, 0, 1, 1);
}

VariableBlock<const VariableMatrix> VariableMatrix::operator()(int row) const {
  return Block(row, 0, 1, 1);
}

VariableBlock<VariableMatrix> VariableMatrix::Block(int rowOffset,
                                                    int colOffset,
                                                    int blockRows,
                                                    int blockCols) {
  return VariableBlock{*this, rowOffset, colOffset, blockRows, blockCols};
}

const VariableBlock<const VariableMatrix> VariableMatrix::Block(
    int rowOffset, int colOffset, int blockRows, int blockCols) const {
  return VariableBlock{*this, rowOffset, colOffset, blockRows, blockCols};
}

VariableBlock<VariableMatrix> VariableMatrix::Segment(int offset, int length) {
  return Block(offset, 0, length, 1);
}

const VariableBlock<const VariableMatrix> VariableMatrix::Segment(
    int offset, int length) const {
  return Block(offset, 0, length, 1);
}

VariableBlock<VariableMatrix> VariableMatrix::Row(int row) {
  return Block(row, 0, 1, Cols());
}

VariableBlock<const VariableMatrix> VariableMatrix::Row(int row) const {
  return Block(row, 0, 1, Cols());
}

VariableBlock<VariableMatrix> VariableMatrix::Col(int col) {
  return Block(0, col, Rows(), 1);
}

VariableBlock<const VariableMatrix> VariableMatrix::Col(int col) const {
  return Block(0, col, Rows(), 1);
}

SLEIPNIR_DLLEXPORT VariableMatrix operator*(const VariableMatrix& lhs,
                                            const VariableMatrix& rhs) {
  assert(lhs.Cols() == rhs.Rows());

  VariableMatrix result{lhs.Rows(), rhs.Cols()};

  for (int i = 0; i < lhs.Rows(); ++i) {
    for (int j = 0; j < rhs.Cols(); ++j) {
      autodiff::Variable sum = 0.0;
      for (int k = 0; k < lhs.Cols(); ++k) {
        sum += lhs.Autodiff(i, k) * rhs.Autodiff(k, j);
      }
      result.Autodiff(i, j) = sum;
    }
  }

  return result;
}

SLEIPNIR_DLLEXPORT VariableMatrix operator*(const VariableMatrix& lhs,
                                            double rhs) {
  VariableMatrix result{lhs.Rows(), lhs.Cols()};

  autodiff::Variable rhsVar{autodiff::MakeConstant(rhs)};
  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) = lhs.Autodiff(row, col) * rhsVar;
    }
  }

  return result;
}

SLEIPNIR_DLLEXPORT VariableMatrix operator*(double lhs,
                                            const VariableMatrix& rhs) {
  VariableMatrix result{rhs.Rows(), rhs.Cols()};

  autodiff::Variable lhsVar{autodiff::MakeConstant(lhs)};
  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) = rhs.Autodiff(row, col) * lhsVar;
    }
  }

  return result;
}

VariableMatrix& VariableMatrix::operator*=(const VariableMatrix& rhs) {
  assert(Cols() == rhs.Rows() && Cols() == rhs.Cols());

  for (int i = 0; i < Rows(); ++i) {
    for (int j = 0; j < rhs.Cols(); ++j) {
      autodiff::Variable sum = 0.0;
      for (int k = 0; k < Cols(); ++k) {
        sum += Autodiff(i, k) * rhs.Autodiff(k, j);
      }
      Autodiff(i, j) = sum;
    }
  }

  return *this;
}

VariableMatrix& VariableMatrix::operator*=(double rhs) {
  for (int row = 0; row < Rows(); ++row) {
    for (int col = 0; col < Cols(); ++col) {
      Autodiff(row, col) *= autodiff::Variable{autodiff::MakeConstant(rhs)};
    }
  }

  return *this;
}

SLEIPNIR_DLLEXPORT VariableMatrix operator/(const VariableMatrix& lhs,
                                            const VariableMatrix& rhs) {
  assert(rhs.Rows() == 1 && rhs.Cols() == 1);

  VariableMatrix result{lhs.Rows(), lhs.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) = lhs.Autodiff(row, col) / rhs.Autodiff(0, 0);
    }
  }

  return result;
}

SLEIPNIR_DLLEXPORT VariableMatrix operator/(const VariableMatrix& lhs,
                                            double rhs) {
  VariableMatrix result{lhs.Rows(), lhs.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) =
          lhs.Autodiff(row, col) /
          autodiff::Variable{autodiff::MakeConstant(rhs)};
    }
  }

  return result;
}

VariableMatrix& VariableMatrix::operator/=(const VariableMatrix& rhs) {
  for (int row = 0; row < Rows(); ++row) {
    for (int col = 0; col < Cols(); ++col) {
      Autodiff(row, col) /= rhs.Autodiff(0, 0);
    }
  }

  return *this;
}

VariableMatrix& VariableMatrix::operator/=(double rhs) {
  for (int row = 0; row < Rows(); ++row) {
    for (int col = 0; col < Cols(); ++col) {
      Autodiff(row, col) /= autodiff::Variable{autodiff::MakeConstant(rhs)};
    }
  }

  return *this;
}

SLEIPNIR_DLLEXPORT VariableMatrix operator+(const VariableMatrix& lhs,
                                            const VariableMatrix& rhs) {
  VariableMatrix result{lhs.Rows(), lhs.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) =
          lhs.Autodiff(row, col) + rhs.Autodiff(row, col);
    }
  }

  return result;
}

VariableMatrix& VariableMatrix::operator+=(const VariableMatrix& rhs) {
  for (int row = 0; row < Rows(); ++row) {
    for (int col = 0; col < Cols(); ++col) {
      Autodiff(row, col) += rhs.Autodiff(row, col);
    }
  }

  return *this;
}

SLEIPNIR_DLLEXPORT VariableMatrix operator-(const VariableMatrix& lhs,
                                            const VariableMatrix& rhs) {
  VariableMatrix result{lhs.Rows(), lhs.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) =
          lhs.Autodiff(row, col) - rhs.Autodiff(row, col);
    }
  }

  return result;
}

VariableMatrix& VariableMatrix::operator-=(const VariableMatrix& rhs) {
  for (int row = 0; row < Rows(); ++row) {
    for (int col = 0; col < Cols(); ++col) {
      Autodiff(row, col) -= rhs.Autodiff(row, col);
    }
  }

  return *this;
}

SLEIPNIR_DLLEXPORT VariableMatrix operator-(const VariableMatrix& lhs) {
  VariableMatrix result{lhs.Rows(), lhs.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) = -lhs.Autodiff(row, col);
    }
  }

  return result;
}

VariableMatrix VariableMatrix::Transpose() const {
  VariableMatrix result{Cols(), Rows()};

  for (int row = 0; row < Rows(); ++row) {
    for (int col = 0; col < Cols(); ++col) {
      result.Autodiff(col, row) = Autodiff(row, col);
    }
  }

  return result;
}

int VariableMatrix::Rows() const {
  return m_rows;
}

int VariableMatrix::Cols() const {
  return m_cols;
}

double VariableMatrix::Value(int row, int col) const {
  return m_storage[row * Cols() + col].Value();
}

double VariableMatrix::Value(int index) const {
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

autodiff::Variable& VariableMatrix::Autodiff(int row, int col) {
  return m_storage[row * Cols() + col];
}

const autodiff::Variable& VariableMatrix::Autodiff(int row, int col) const {
  return m_storage[row * Cols() + col];
}

VariableMatrix abs(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) = autodiff::abs(x.Autodiff(row, col));
    }
  }

  return result;
}

VariableMatrix acos(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) = autodiff::acos(x.Autodiff(row, col));
    }
  }

  return result;
}

VariableMatrix asin(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) = autodiff::asin(x.Autodiff(row, col));
    }
  }

  return result;
}

VariableMatrix atan(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) = autodiff::atan(x.Autodiff(row, col));
    }
  }

  return result;
}

VariableMatrix atan2(const VariableMatrix& y, const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) =
          autodiff::atan2(y.Autodiff(row, col), x.Autodiff(row, col));
    }
  }

  return result;
}

VariableMatrix cos(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) = autodiff::cos(x.Autodiff(row, col));
    }
  }

  return result;
}

VariableMatrix cosh(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) = autodiff::cosh(x.Autodiff(row, col));
    }
  }

  return result;
}

VariableMatrix erf(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) = autodiff::erf(x.Autodiff(row, col));
    }
  }

  return result;
}

VariableMatrix exp(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) = autodiff::exp(x.Autodiff(row, col));
    }
  }

  return result;
}

VariableMatrix hypot(const VariableMatrix& x, const VariableMatrix& y) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) =
          autodiff::hypot(x.Autodiff(row, col), y.Autodiff(row, col));
    }
  }

  return result;
}

VariableMatrix log(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) = autodiff::log(x.Autodiff(row, col));
    }
  }

  return result;
}

VariableMatrix log10(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) = autodiff::log10(x.Autodiff(row, col));
    }
  }

  return result;
}

VariableMatrix pow(double base, const VariableMatrix& power) {
  VariableMatrix result{power.Rows(), power.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) =
          autodiff::pow(autodiff::Variable{autodiff::MakeConstant(base)},
                        power.Autodiff(row, col));
    }
  }

  return result;
}

VariableMatrix pow(const VariableMatrix& base, double power) {
  VariableMatrix result{base.Rows(), base.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) =
          autodiff::pow(base.Autodiff(row, col),
                        autodiff::Variable{autodiff::MakeConstant(power)});
    }
  }

  return result;
}

VariableMatrix pow(const VariableMatrix& base, const VariableMatrix& power) {
  VariableMatrix result{base.Rows(), base.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) =
          autodiff::pow(base.Autodiff(row, col), power.Autodiff(row, col));
    }
  }

  return result;
}

VariableMatrix sin(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) = autodiff::sin(x.Autodiff(row, col));
    }
  }

  return result;
}

VariableMatrix sinh(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) = autodiff::sinh(x.Autodiff(row, col));
    }
  }

  return result;
}

VariableMatrix sqrt(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) = autodiff::sqrt(x.Autodiff(row, col));
    }
  }

  return result;
}

VariableMatrix tan(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) = autodiff::tan(x.Autodiff(row, col));
    }
  }

  return result;
}

VariableMatrix tanh(const VariableMatrix& x) {
  VariableMatrix result{x.Rows(), x.Cols()};

  for (int row = 0; row < result.Rows(); ++row) {
    for (int col = 0; col < result.Cols(); ++col) {
      result.Autodiff(row, col) = autodiff::tanh(x.Autodiff(row, col));
    }
  }

  return result;
}

}  // namespace sleipnir
