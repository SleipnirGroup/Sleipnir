// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/SparseUtil.hpp"

#include <functional>

namespace sleipnir {

Eigen::SparseMatrix<double> SparseDiagonal(const Eigen::VectorXd& src) {
  std::vector<Eigen::Triplet<double>> triplets;

  for (int row = 0; row < src.rows(); ++row) {
    triplets.emplace_back(row, row, src(row));
  }

  Eigen::SparseMatrix<double> dest{src.rows(), src.rows()};
  dest.setFromTriplets(triplets.begin(), triplets.end());

  return dest;
}

Eigen::SparseMatrix<double> SparseIdentity(int rows, int cols) {
  std::vector<Eigen::Triplet<double>> triplets;

  for (int row = 0; row < std::min(rows, cols); ++row) {
    triplets.emplace_back(row, row, 1.0);
  }

  Eigen::SparseMatrix<double> dest{rows, cols};
  dest.setFromTriplets(triplets.begin(), triplets.end());

  return dest;
}

void AssignSparseBlock(std::vector<Eigen::Triplet<double>>& triplets,
                       int rowOffset, int colOffset,
                       const Eigen::SparseMatrix<double>& mat) {
  for (int k = 0; k < mat.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it{mat, k}; it; ++it) {
      triplets.emplace_back(rowOffset + it.row(), colOffset + it.col(),
                            it.value());
    }
  }
}

void Spy(std::string_view filename, const Eigen::SparseMatrix<double>& mat) {
  std::ofstream file{std::string{filename}};
  if (!file.is_open()) {
    return;
  }

  Spy(file, mat);
}

void Spy(std::ostream& file, const Eigen::SparseMatrix<double>& mat) {
  const int cells_width = mat.cols() + 1;
  const int cells_height = mat.rows();

  std::vector<uint8_t> cells;

  // Allocate space for matrix of characters plus trailing newlines
  cells.reserve(cells_width * cells_height);

  // Initialize cell array
  for (int row = 0; row < mat.rows(); ++row) {
    for (int col = 0; col < mat.cols(); ++col) {
      cells.emplace_back('.');
    }
    cells.emplace_back('\n');
  }

  // Fill in non-sparse entries
  for (int k = 0; k < mat.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it{mat, k}; it; ++it) {
      if (it.value() < 0.0) {
        cells[it.row() * cells_width + it.col()] = '-';
      } else if (it.value() > 0.0) {
        cells[it.row() * cells_width + it.col()] = '+';
      }
    }
  }

  // Write cell array to file
  for (const auto& c : cells) {
    file << c;
  }
}

}  // namespace sleipnir
