// Copyright (c) Joshua Nichols and Tyler Veness

#include "sleipnir/optimization/Constraints.hpp"

#include <cassert>

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
std::vector<autodiff::Variable> MakeConstraints(const VariableMatrix& lhs,
                                                const VariableMatrix& rhs) {
  std::vector<autodiff::Variable> constraints;

  if (lhs.Rows() == 1 && lhs.Cols() == 1) {
    constraints.reserve(rhs.Rows() * rhs.Cols());

    for (int row = 0; row < rhs.Rows(); ++row) {
      for (int col = 0; col < rhs.Cols(); ++col) {
        // Make right-hand side zero
        constraints.emplace_back(lhs.Autodiff(0, 0) - rhs.Autodiff(row, col));
      }
    }
  } else if (rhs.Rows() == 1 && rhs.Cols() == 1) {
    constraints.reserve(lhs.Rows() * lhs.Cols());

    for (int row = 0; row < lhs.Rows(); ++row) {
      for (int col = 0; col < lhs.Cols(); ++col) {
        // Make right-hand side zero
        constraints.emplace_back(lhs.Autodiff(row, col) - rhs.Autodiff(0, 0));
      }
    }
  } else {
    assert(lhs.Rows() == rhs.Rows() && lhs.Cols() == rhs.Cols());
    constraints.reserve(lhs.Rows() * lhs.Cols());

    for (int row = 0; row < lhs.Rows(); ++row) {
      for (int col = 0; col < lhs.Cols(); ++col) {
        // Make right-hand side zero
        constraints.emplace_back(lhs.Autodiff(row, col) -
                                 rhs.Autodiff(row, col));
      }
    }
  }

  return constraints;
}

}  // namespace

EqualityConstraints operator==(const VariableMatrix& lhs,
                               const VariableMatrix& rhs) {
  return EqualityConstraints{MakeConstraints(lhs, rhs)};
}

InequalityConstraints operator<(const VariableMatrix& lhs,
                                const VariableMatrix& rhs) {
  // Since the solver can make lhs arbitrarily close to rhs, just use <=
  return lhs <= rhs;
}

InequalityConstraints operator<=(const VariableMatrix& lhs,
                                 const VariableMatrix& rhs) {
  return rhs >= lhs;
}

InequalityConstraints operator>(const VariableMatrix& lhs,
                                const VariableMatrix& rhs) {
  // Since the solver can make lhs arbitrarily close to rhs, just use >=
  return lhs >= rhs;
}

InequalityConstraints operator>=(const VariableMatrix& lhs,
                                 const VariableMatrix& rhs) {
  return InequalityConstraints{MakeConstraints(lhs, rhs)};
}

}  // namespace sleipnir
