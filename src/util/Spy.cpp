// Copyright (c) Sleipnir contributors

#include "sleipnir/util/Spy.hpp"

#include <fstream>

namespace sleipnir {

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
