// Copyright (c) Sleipnir contributors

#pragma once

#include <cstddef>

#include <Eigen/Core>

namespace slp {

/**
 * Represents the inertia of a matrix (the number of positive, negative, and
 * zero eigenvalues).
 */
class Inertia {
 public:
  size_t positive = 0;
  size_t negative = 0;
  size_t zero = 0;

  constexpr Inertia() = default;

  /**
   * Constructs Inertia with the given number of positive, negative, and zero
   * eigenvalues.
   *
   * @param positive The number of positive eigenvalues.
   * @param negative The number of negative eigenvalues.
   * @param zero The number of zero eigenvalues.
   */
  constexpr Inertia(size_t positive, size_t negative, size_t zero)
      : positive{positive}, negative{negative}, zero{zero} {}

  /**
   * Constructs Inertia from the D matrix of an LDLT decomposition
   * (see https://en.wikipedia.org/wiki/Sylvester's_law_of_inertia).
   *
   * @param D The D matrix of an LDLT decomposition in vector form.
   */
  explicit Inertia(const Eigen::VectorXd& D) {
    for (const auto& elem : D) {
      if (elem > 0.0) {
        ++positive;
      } else if (elem < 0.0) {
        ++negative;
      } else {
        ++zero;
      }
    }
  }

  bool operator==(const Inertia&) const = default;
};

}  // namespace slp
