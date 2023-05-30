// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <vector>

#include <Eigen/Core>

#include "sleipnir/autodiff/Variable.hpp"

namespace sleipnir {

/**
 * Filter entry consisting of cost and constraint violation.
 */
struct FilterEntry {
  /// The cost function's value
  double cost = 0.0;

  /// The constraint violation
  double constraintViolation = 0.0;

  constexpr FilterEntry() = default;

  /**
   * Constructs a FilterEntry.
   *
   * @param f The cost function.
   * @param mu The barrier parameter.
   * @param s The inequality constraint slack variables.
   * @param c_e The equality constraint values (nonzero means violation).
   * @param c_i The inequality constraint values (negative means violation).
   */
  FilterEntry(const Variable& f, double mu, Eigen::VectorXd& s,
              const Eigen::VectorXd& c_e, const Eigen::VectorXd& c_i)
      : cost{f.Value() - mu * s.array().log().sum()},
        constraintViolation{c_e.lpNorm<1>() + (c_i - s).lpNorm<1>()} {}
};

/**
 * Interior-point step filter.
 */
class Filter {
 public:
  /**
   * Initialize the filter with an entry.
   *
   * @param entry The initial filter entry.
   */
  explicit Filter(const FilterEntry& entry) {
    m_filter.push_back(entry);
    m_maxConstraintViolation = 1e4 * std::max(1.0, entry.constraintViolation);
  }

  /**
   * Initialize the filter with an entry.
   *
   * @param entry The initial filter entry.
   */
  explicit Filter(FilterEntry&& entry) {
    m_filter.push_back(entry);
    m_maxConstraintViolation = 1e4 * std::max(1.0, entry.constraintViolation);
  }

  /**
   * Add a new entry to the filter.
   *
   * @param entry The entry to add to the filter.
   */
  void Add(const FilterEntry& entry) { m_filter.push_back(entry); }

  /**
   * Add a new entry to the filter.
   *
   * @param entry The entry to add to the filter.
   */
  void Add(FilterEntry&& entry) { m_filter.push_back(entry); }

  /**
   * Returns the last entry in the filter.
   */
  const FilterEntry& LastEntry() const { return m_filter.back(); }

  /**
   * Reset the filter with an entry.
   *
   * @param entry The initial filter entry.
   */
  void Reset(const FilterEntry& entry) {
    m_filter.clear();
    m_filter.push_back(entry);
  }

  /**
   * Reset the filter with an entry.
   *
   * @param entry The initial filter entry.
   */
  void Reset(FilterEntry&& entry) {
    m_filter.clear();
    m_filter.push_back(entry);
  }

  /**
   * Returns true if the given entry is acceptable to the filter.
   *
   * @param entry The entry to check.
   */
  bool IsAcceptable(const FilterEntry& entry) {
    // If current filter entry is better than all prior ones in some respect,
    // accept it
    return std::all_of(m_filter.begin(), m_filter.end(), [&](const auto& elem) {
      return entry.cost <= elem.cost - kGammaCost * elem.constraintViolation ||
             entry.constraintViolation <=
                 (1.0 - kGammaConstraint) * elem.constraintViolation;
    });
  }

 private:
  std::vector<FilterEntry> m_filter;

  double m_maxConstraintViolation = 1e4;

  static constexpr double kGammaCost = 1e-8;
  static constexpr double kGammaConstraint = 1e-5;
};

}  // namespace sleipnir
