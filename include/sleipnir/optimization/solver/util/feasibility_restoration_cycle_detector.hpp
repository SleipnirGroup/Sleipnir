// Copyright (c) Sleipnir contributors

#pragma once

#include <limits>

namespace slp {

/// Detects cycling between the normal phase and feasibility restoration.
///
/// Feasibility restoration normally stops once it sufficiently reduces the
/// constraint violation, but the normal phase can then drive the constraint
/// violation back up, so the solver cycles between them. On infeasible
/// problems, this prevents feasibility restoration from ever converging to a
/// minimizer of the constraint violation, which is the certificate of local
/// infeasibility.
///
/// @tparam Scalar Scalar type.
template <typename Scalar>
class FeasibilityRestorationCycleDetector {
 public:
  /// Records an entry into feasibility restoration.
  ///
  /// @param constraint_violation The constraint violation at entry.
  void record_entry(Scalar constraint_violation) {
    ++m_num_entries;
    if (constraint_violation < Scalar(0.9) * m_min_entry_violation) {
      m_min_entry_violation = constraint_violation;
      m_num_stalled_entries = 0;
    } else {
      ++m_num_stalled_entries;
    }
  }

  /// Returns true if the solver is cycling, in which case feasibility
  /// restoration should be run to convergence.
  ///
  /// The solver is cycling if several consecutive entries didn't improve on the
  /// smallest constraint violation at entry.
  ///
  /// This is never true during the first few entries. Early entries often fail
  /// to improve while the solver is still far from a feasible region, and
  /// converging feasibility restoration there can find a local minimizer of the
  /// constraint violation even though the problem is feasible.
  ///
  /// @return True if the solver is cycling.
  bool is_cycling() const {
    return m_num_entries > 5 && m_num_stalled_entries >= 2;
  }

 private:
  int m_num_entries = 0;
  Scalar m_min_entry_violation = std::numeric_limits<Scalar>::infinity();
  int m_num_stalled_entries = 0;
};

}  // namespace slp
