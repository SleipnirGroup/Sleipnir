// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/Constraints.hpp"

#include <algorithm>

namespace sleipnir {

namespace {

/**
 * Make a list of constraints.
 *
 * The standard form for equality constraints is c(x) = 0, and the standard form
 * for inequality constraints is c(x) ≥ 0. This function takes constraints of
 * the form lhs = rhs or lhs ≥ rhs and converts them to lhs - rhs = 0 or
 * lhs - rhs ≥ 0.
 */
std::vector<Variable> MakeConstraints(const VariableMatrix& lhs,
                                      const VariableMatrix& rhs) {
  std::vector<Variable> constraints;

  if (lhs.Rows() == 1 && lhs.Cols() == 1) {
    constraints.reserve(rhs.Rows() * rhs.Cols());

    for (int row = 0; row < rhs.Rows(); ++row) {
      for (int col = 0; col < rhs.Cols(); ++col) {
        // Make right-hand side zero
        constraints.emplace_back(lhs(0, 0) - rhs(row, col));
      }
    }
  } else if (rhs.Rows() == 1 && rhs.Cols() == 1) {
    constraints.reserve(lhs.Rows() * lhs.Cols());

    for (int row = 0; row < lhs.Rows(); ++row) {
      for (int col = 0; col < lhs.Cols(); ++col) {
        // Make right-hand side zero
        constraints.emplace_back(lhs(row, col) - rhs(0, 0));
      }
    }
  } else {
    assert(lhs.Rows() == rhs.Rows() && lhs.Cols() == rhs.Cols());
    constraints.reserve(lhs.Rows() * lhs.Cols());

    for (int row = 0; row < lhs.Rows(); ++row) {
      for (int col = 0; col < lhs.Cols(); ++col) {
        // Make right-hand side zero
        constraints.emplace_back(lhs(row, col) - rhs(row, col));
      }
    }
  }

  return constraints;
}

}  // namespace

EqualityConstraints::EqualityConstraints(const Variable& lhs,
                                         const Variable& rhs)
    : constraints{{lhs - rhs}} {}

EqualityConstraints::EqualityConstraints(const VariableMatrix& lhs,
                                         const VariableMatrix& rhs)
    : constraints{MakeConstraints(lhs, rhs)} {}

EqualityConstraints::operator bool() const {
  return std::all_of(
      constraints.begin(), constraints.end(),
      [](const auto& constraint) { return constraint.Value() == 0.0; });
}

InequalityConstraints::InequalityConstraints(const Variable& lhs,
                                             const Variable& rhs)
    : constraints{{lhs - rhs}} {}

InequalityConstraints::InequalityConstraints(const VariableMatrix& lhs,
                                             const VariableMatrix& rhs)
    : constraints{MakeConstraints(lhs, rhs)} {}

InequalityConstraints::operator bool() const {
  return std::all_of(
      constraints.begin(), constraints.end(),
      [](const auto& constraint) { return constraint.Value() >= 0.0; });
}

EqualityConstraints operator==(const Variable& lhs, const Variable& rhs) {
  return EqualityConstraints{lhs, rhs};
}

InequalityConstraints operator<(const Variable& lhs, const Variable& rhs) {
  return rhs >= lhs;
}

InequalityConstraints operator<=(const Variable& lhs, const Variable& rhs) {
  return rhs >= lhs;
}

InequalityConstraints operator>(const Variable& lhs, const Variable& rhs) {
  return lhs >= rhs;
}

InequalityConstraints operator>=(const Variable& lhs, const Variable& rhs) {
  return InequalityConstraints{lhs, rhs};
}

EqualityConstraints operator==(const VariableMatrix& lhs,
                               const VariableMatrix& rhs) {
  return EqualityConstraints{lhs, rhs};
}

InequalityConstraints operator<(const VariableMatrix& lhs,
                                const VariableMatrix& rhs) {
  return lhs <= rhs;
}

InequalityConstraints operator<=(const VariableMatrix& lhs,
                                 const VariableMatrix& rhs) {
  return rhs >= lhs;
}

InequalityConstraints operator>(const VariableMatrix& lhs,
                                const VariableMatrix& rhs) {
  return lhs >= rhs;
}

InequalityConstraints operator>=(const VariableMatrix& lhs,
                                 const VariableMatrix& rhs) {
  return InequalityConstraints{lhs, rhs};
}

}  // namespace sleipnir
