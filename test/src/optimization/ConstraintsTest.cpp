// Copyright (c) Sleipnir contributors

#include <array>
#include <tuple>

#include <gtest/gtest.h>
#include <sleipnir/autodiff/Variable.hpp>
#include <sleipnir/autodiff/VariableMatrix.hpp>
#include <sleipnir/optimization/Constraints.hpp>

TEST(ConstraintsTest, EqualityConstraintBooleanComparisons) {
  using sleipnir::Variable;
  using sleipnir::VariableMatrix;

  constexpr std::array<std::tuple<double, double>, 3> args{
      std::tuple{1.0, 1.0}, {1.0, 2.0}, {2.0, 1.0}};

  // double-Variable
  for (const auto& [lhs, rhs] : args) {
    EXPECT_EQ(double{lhs} == Variable{rhs}, lhs == rhs);
  }

  // double-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    EXPECT_EQ(double{lhs} == VariableMatrix{{rhs}}, lhs == rhs);
  }

  // Variable-double
  for (const auto& [lhs, rhs] : args) {
    EXPECT_EQ(Variable{lhs} == double{rhs}, lhs == rhs);
  }

  // Variable-Variable
  for (const auto& [lhs, rhs] : args) {
    EXPECT_EQ(Variable{lhs} == Variable{rhs}, lhs == rhs);
  }

  // Variable-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    EXPECT_EQ(Variable{lhs} == VariableMatrix{{rhs}}, lhs == rhs);
  }

  // VariableMatrix-double
  for (const auto& [lhs, rhs] : args) {
    EXPECT_EQ(VariableMatrix{{lhs}} == double{rhs}, lhs == rhs);
  }

  // VariableMatrix-Variable
  for (const auto& [lhs, rhs] : args) {
    EXPECT_EQ(VariableMatrix{{lhs}} == Variable{rhs}, lhs == rhs);
  }

  // VariableMatrix-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    EXPECT_EQ(VariableMatrix{{lhs}} == VariableMatrix{{rhs}}, lhs == rhs);
  }
}

// For the purposes of optimization, a < constraint is treated the same as a <=
// constraint
TEST(ConstraintsTest, InequalityConstraintBooleanComparisons) {
  using sleipnir::Variable;
  using sleipnir::VariableMatrix;

  constexpr std::array<std::tuple<double, double>, 3> args{
      std::tuple{1.0, 1.0}, {1.0, 2.0}, {2.0, 1.0}};

  // double-Variable
  for (const auto& [lhs, rhs] : args) {
    EXPECT_EQ(double{lhs} < Variable{rhs}, lhs <= rhs);
    EXPECT_EQ(double{lhs} <= Variable{rhs}, lhs <= rhs);
    EXPECT_EQ(double{lhs} > Variable{rhs}, lhs >= rhs);
    EXPECT_EQ(double{lhs} >= Variable{rhs}, lhs >= rhs);
  }

  // double-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    EXPECT_EQ(double{lhs} < VariableMatrix{{rhs}}, lhs <= rhs);
    EXPECT_EQ(double{lhs} <= VariableMatrix{{rhs}}, lhs <= rhs);
    EXPECT_EQ(double{lhs} > VariableMatrix{{rhs}}, lhs >= rhs);
    EXPECT_EQ(double{lhs} >= VariableMatrix{{rhs}}, lhs >= rhs);
  }

  // Variable-double
  for (const auto& [lhs, rhs] : args) {
    EXPECT_EQ(Variable{lhs} < double{rhs}, lhs <= rhs);
    EXPECT_EQ(Variable{lhs} <= double{rhs}, lhs <= rhs);
    EXPECT_EQ(Variable{lhs} > double{rhs}, lhs >= rhs);
    EXPECT_EQ(Variable{lhs} >= double{rhs}, lhs >= rhs);
  }

  // Variable-Variable
  for (const auto& [lhs, rhs] : args) {
    EXPECT_EQ(Variable{lhs} < Variable{rhs}, lhs <= rhs);
    EXPECT_EQ(Variable{lhs} <= Variable{rhs}, lhs <= rhs);
    EXPECT_EQ(Variable{lhs} > Variable{rhs}, lhs >= rhs);
    EXPECT_EQ(Variable{lhs} >= Variable{rhs}, lhs >= rhs);
  }

  // Variable-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    EXPECT_EQ(Variable{lhs} < VariableMatrix{{rhs}}, lhs <= rhs);
    EXPECT_EQ(Variable{lhs} <= VariableMatrix{{rhs}}, lhs <= rhs);
    EXPECT_EQ(Variable{lhs} > VariableMatrix{{rhs}}, lhs >= rhs);
    EXPECT_EQ(Variable{lhs} >= VariableMatrix{{rhs}}, lhs >= rhs);
  }

  // VariableMatrix-double
  for (const auto& [lhs, rhs] : args) {
    EXPECT_EQ(VariableMatrix{{lhs}} < double{rhs}, lhs <= rhs);
    EXPECT_EQ(VariableMatrix{{lhs}} <= double{rhs}, lhs <= rhs);
    EXPECT_EQ(VariableMatrix{{lhs}} > double{rhs}, lhs >= rhs);
    EXPECT_EQ(VariableMatrix{{lhs}} >= double{rhs}, lhs >= rhs);
  }

  // VariableMatrix-Variable
  for (const auto& [lhs, rhs] : args) {
    EXPECT_EQ(VariableMatrix{{lhs}} < Variable{rhs}, lhs <= rhs);
    EXPECT_EQ(VariableMatrix{{lhs}} <= Variable{rhs}, lhs <= rhs);
    EXPECT_EQ(VariableMatrix{{lhs}} > Variable{rhs}, lhs >= rhs);
    EXPECT_EQ(VariableMatrix{{lhs}} >= Variable{rhs}, lhs >= rhs);
  }

  // VariableMatrix-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    EXPECT_EQ(VariableMatrix{{lhs}} < VariableMatrix{{rhs}}, lhs <= rhs);
    EXPECT_EQ(VariableMatrix{{lhs}} <= VariableMatrix{{rhs}}, lhs <= rhs);
    EXPECT_EQ(VariableMatrix{{lhs}} > VariableMatrix{{rhs}}, lhs >= rhs);
    EXPECT_EQ(VariableMatrix{{lhs}} >= VariableMatrix{{rhs}}, lhs >= rhs);
  }
}
