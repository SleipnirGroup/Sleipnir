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
 * Returns sparse identity matrix.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 */
Eigen::SparseMatrix<double> SparseIdentity(int rows, int cols);

/**
 * Adds a sparse matrix to the list of triplets with the given row and column
 * offset.
 *
 * @param[out] triplets The triplet storage.
 * @param[in] rowOffset The row offset for each triplet.
 * @param[in] colOffset The column offset for each triplet.
 * @param[in] mat The matrix to iterate over.
 */
void AssignSparseBlock(std::vector<Eigen::Triplet<double>>& triplets,
                       int rowOffset, int colOffset,
                       const Eigen::SparseMatrix<double>& mat);

}  // namespace sleipnir
