// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/SparseUtil.hpp"

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

}  // namespace sleipnir
