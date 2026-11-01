// Copyright (c) Sleipnir contributors

#pragma once

#include <format>

namespace slp {

/// Represents the inertia of a matrix (the number of positive, negative, and
/// zero eigenvalues).
class Inertia {
 public:
  /// The number of positive eigenvalues.
  int positive = 0;
  /// The number of negative eigenvalues.
  int negative = 0;
  /// The number of zero eigenvalues.
  int zero = 0;

  constexpr Inertia() = default;

  /// Constructs Inertia with the given number of positive, negative, and zero
  /// eigenvalues.
  ///
  /// @param positive The number of positive eigenvalues.
  /// @param negative The number of negative eigenvalues.
  /// @param zero The number of zero eigenvalues.
  constexpr Inertia(int positive, int negative, int zero)
      : positive{positive}, negative{negative}, zero{zero} {}

  /// Inertia equality operator.
  ///
  /// @return True if Inertia is equal.
  bool operator==(const Inertia&) const = default;
};

}  // namespace slp

/// Formatter for Inertia.
template <>
struct std::formatter<slp::Inertia> {
  /// Parses format string.
  ///
  /// @param ctx Format parse context.
  /// @return Format parse context iterator.
  constexpr auto parse(std::format_parse_context& ctx) {
    return m_underlying.parse(ctx);
  }

  /// Formats Inertia.
  ///
  /// @tparam FmtContext Format context type.
  /// @param inertia Inertia.
  /// @param ctx Format context.
  /// @return Format context iterator.
  template <typename FmtContext>
  auto format(const slp::Inertia& inertia, FmtContext& ctx) const {
    auto out = ctx.out();

    out = std::format_to(out, "(");
    out = m_underlying.format(inertia.positive, ctx);
    out = std::format_to(out, ", ");
    out = m_underlying.format(inertia.negative, ctx);
    out = std::format_to(out, ", ");
    out = m_underlying.format(inertia.zero, ctx);
    return std::format_to(out, ")");
  }

 private:
  std::formatter<int> m_underlying;
};
