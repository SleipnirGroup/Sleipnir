// Copyright (c) Sleipnir contributors

#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace sleipnir {

/**
 * Converts dense column vector into sparse diagonal matrix.
 *
 * @param src Column vector.
 */
Eigen::SparseMatrix<double> SparseDiagonal(const Eigen::VectorXd& src);

/**
 * Adds a sparse matrix to the list of triplets with the given row and column
 * offset.
 *
 * @param[out] triplets The triplet storage.
 * @param[in] rowOffset The row offset for each triplet.
 * @param[in] colOffset The column offset for each triplet.
 * @param[in] mat The matrix to iterate over.
 * @param[in] transpose Whether to transpose mat.
 */
void AssignSparseBlock(std::vector<Eigen::Triplet<double>>& triplets,
                       int rowOffset, int colOffset,
                       const Eigen::SparseMatrix<double>& mat,
                       bool transpose = false);

}  // namespace sleipnir
