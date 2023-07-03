// Copyright (c) Sleipnir contributors

#pragma once

#include <concepts>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <fmt/format.h>

/**
 * Formatter for classes derived from Eigen::MatrixBase<Derived> or
 * Eigen::SparseCompressedBase<Derived>.
 */
template <typename Derived, typename CharT>
  requires std::derived_from<Derived, Eigen::MatrixBase<Derived>> ||
           std::derived_from<Derived, Eigen::SparseCompressedBase<Derived>>
struct fmt::formatter<Derived, CharT> {
  constexpr auto parse(fmt::format_parse_context& ctx) {
    return m_underlying.parse(ctx);
  }

  auto format(const Derived& mat, fmt::format_context& ctx) const {
    auto out = ctx.out();

    for (int row = 0; row < mat.rows(); ++row) {
      for (int col = 0; col < mat.cols(); ++col) {
        out = fmt::format_to(out, "  ");
        out = m_underlying.format(mat.coeff(row, col), ctx);
      }

      if (row < mat.rows() - 1) {
        out = fmt::format_to(out, "\n");
      }
    }

    return out;
  }

 private:
  fmt::formatter<typename Derived::Scalar, CharT> m_underlying;
};
