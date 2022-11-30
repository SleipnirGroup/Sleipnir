// Copyright (c) Joshua Nichols and Tyler Veness

#include "sleipnir/optimization/Constraints.hpp"

#include <cassert>

#include "sleipnir/autodiff/Variable.hpp"
#include "sleipnir/autodiff/VariableMatrix.hpp"

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

EqualityConstraints operator==(const Variable& lhs, const Variable& rhs) {
  return EqualityConstraints{MakeConstraints(lhs, rhs)};
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
  return InequalityConstraints{MakeConstraints(lhs, rhs)};
}

EqualityConstraints operator==(const VariableMatrix& lhs,
                               const VariableMatrix& rhs) {
  return EqualityConstraints{MakeConstraints(lhs, rhs)};
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
  return InequalityConstraints{MakeConstraints(lhs, rhs)};
}

}  // namespace sleipnir
