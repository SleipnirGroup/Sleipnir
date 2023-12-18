// Copyright (c) Sleipnir contributors

#pragma once

#include <iosfwd>
#include <string_view>

#include <Eigen/SparseCore>

#include "sleipnir/util/SymbolExports.hpp"

namespace sleipnir {

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
