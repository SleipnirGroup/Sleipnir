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

void AssignSparseBlock(std::vector<Eigen::Triplet<double>>& triplets,
                       int rowOffset, int colOffset,
                       const Eigen::SparseMatrix<double>& mat, bool transpose) {
  for (int k = 0; k < mat.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it{mat, k}; it; ++it) {
      if (transpose) {
        triplets.emplace_back(rowOffset + it.col(), colOffset + it.row(),
                              it.value());
      } else {
        triplets.emplace_back(rowOffset + it.row(), colOffset + it.col(),
                              it.value());
      }
    }
  }
}

}  // namespace sleipnir
