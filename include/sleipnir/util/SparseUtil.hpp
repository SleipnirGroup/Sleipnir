// Copyright (c) Sleipnir contributors

#pragma once

#include <fstream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "sleipnir/SymbolExports.hpp"

namespace sleipnir {

/**
 * Converts dense column vector into sparse diagonal matrix.
 *
 * @param src Column vector.
 */
SLEIPNIR_DLLEXPORT Eigen::SparseMatrix<double> SparseDiagonal(
    const Eigen::VectorXd& src);

/**
 * Returns sparse identity matrix.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 */
SLEIPNIR_DLLEXPORT Eigen::SparseMatrix<double> SparseIdentity(int rows,
                                                              int cols);

/**
 * Adds a sparse matrix to the list of triplets with the given row and column
 * offset.
 *
 * @param[out] triplets The triplet storage.
 * @param[in] rowOffset The row offset for each triplet.
 * @param[in] colOffset The column offset for each triplet.
 * @param[in] mat The matrix to iterate over.
 */
SLEIPNIR_DLLEXPORT void AssignSparseBlock(
    std::vector<Eigen::Triplet<double>>& triplets, int rowOffset, int colOffset,
    const Eigen::SparseMatrix<double>& mat);

/**
 * Write the sparsity pattern of a sparse matrix to a file.
 *
 * Each character represents an element with "." representing zero, "+"
 * representing positive, and "-" representing negative. Here's an example for a
 * 3x3 identity matrix.
 *
 * "+.."
 * ".+."
 * "..+"
 *
 * @param[in] filename The filename.
 * @param[in] mat The sparse matrix.
 */
SLEIPNIR_DLLEXPORT void Spy(std::string_view filename,
                            const Eigen::SparseMatrix<double>& mat);

/**
 * Write the sparsity pattern of a sparse matrix to a file.
 *
 * Each character represents an element with '.' representing zero, '+'
 * representing positive, and '-' representing negative. Here's an example for a
 * 3x3 identity matrix.
 *
 * "+.."
 * ".+."
 * "..+"
 *
 * @param[out] file A file stream.
 * @param[in] mat The sparse matrix.
 */
SLEIPNIR_DLLEXPORT void Spy(std::ostream& file,
                            const Eigen::SparseMatrix<double>& mat);

}  // namespace sleipnir
