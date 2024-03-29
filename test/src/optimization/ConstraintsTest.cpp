// Copyright (c) Sleipnir contributors

#include <array>
#include <tuple>

#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/Variable.hpp>
#include <sleipnir/autodiff/VariableMatrix.hpp>
#include <sleipnir/optimization/Constraints.hpp>

#include "CatchStringConverters.hpp"

TEST_CASE("Constraints - Equality constraint boolean comparison",
          "[Constraints]") {
  using sleipnir::Variable;
  using sleipnir::VariableMatrix;

  constexpr std::array<std::tuple<double, double>, 3> args{
      std::tuple{1.0, 1.0}, {1.0, 2.0}, {2.0, 1.0}};

  // double-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK((double{lhs} == Variable{rhs}) == (lhs == rhs));
  }

  // double-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK((double{lhs} == VariableMatrix{{rhs}}) == (lhs == rhs));
  }

  // Variable-double
  for (const auto& [lhs, rhs] : args) {
    CHECK((Variable{lhs} == double{rhs}) == (lhs == rhs));
  }

  // Variable-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK((Variable{lhs} == Variable{rhs}) == (lhs == rhs));
  }

  // Variable-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK((Variable{lhs} == VariableMatrix{{rhs}}) == (lhs == rhs));
  }

  // VariableMatrix-double
  for (const auto& [lhs, rhs] : args) {
    CHECK((VariableMatrix{{lhs}} == double{rhs}) == (lhs == rhs));
  }

  // VariableMatrix-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK((VariableMatrix{{lhs}} == Variable{rhs}) == (lhs == rhs));
  }

  // VariableMatrix-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK((VariableMatrix{{lhs}} == VariableMatrix{{rhs}}) == (lhs == rhs));
  }

  // Eigen::MatrixXd-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK((Eigen::MatrixXd{{lhs}} == Variable{rhs}) == (lhs == rhs));
  }

  // Eigen::MatrixXd-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK((Eigen::MatrixXd{{lhs}} == VariableMatrix{{rhs}}) == (lhs == rhs));
  }

  // Eigen::MatrixXd-VariableBlock
  for (const auto& [lhs, rhs] : args) {
    CHECK((Eigen::MatrixXd{{lhs}} == VariableMatrix{{rhs}}.Block(0, 0, 1, 1)) ==
          (lhs == rhs));
  }

  // Variable-Eigen::MatrixXd
  for (const auto& [lhs, rhs] : args) {
    CHECK((Variable{lhs} == Eigen::MatrixXd{{rhs}}) == (lhs == rhs));
  }

  // VariableMatrix-Eigen::MatrixXd
  for (const auto& [lhs, rhs] : args) {
    CHECK((VariableMatrix{{lhs}} == Eigen::MatrixXd{{rhs}}) == (lhs == rhs));
  }

  // VariableBlock-Eigen::MatrixXd
  for (const auto& [lhs, rhs] : args) {
    CHECK((VariableMatrix{{lhs}}.Block(0, 0, 1, 1) == Eigen::MatrixXd{{rhs}}) ==
          (lhs == rhs));
  }
}

// For the purposes of optimization, a < constraint is treated the same as a <=
// constraint
TEST_CASE("Constraints - Inequality constraint boolean comparisons",
          "[Constraints]") {
  using sleipnir::Variable;
  using sleipnir::VariableMatrix;

  constexpr std::array<std::tuple<double, double>, 3> args{
      std::tuple{1.0, 1.0}, {1.0, 2.0}, {2.0, 1.0}};

  // double-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK((double{lhs} < Variable{rhs}) == (lhs <= rhs));
    CHECK((double{lhs} <= Variable{rhs}) == (lhs <= rhs));
    CHECK((double{lhs} > Variable{rhs}) == (lhs >= rhs));
    CHECK((double{lhs} >= Variable{rhs}) == (lhs >= rhs));
  }

  // double-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK((double{lhs} < VariableMatrix{{rhs}}) == (lhs <= rhs));
    CHECK((double{lhs} <= VariableMatrix{{rhs}}) == (lhs <= rhs));
    CHECK((double{lhs} > VariableMatrix{{rhs}}) == (lhs >= rhs));
    CHECK((double{lhs} >= VariableMatrix{{rhs}}) == (lhs >= rhs));
  }

  // Variable-double
  for (const auto& [lhs, rhs] : args) {
    CHECK((Variable{lhs} < double{rhs}) == (lhs <= rhs));
    CHECK((Variable{lhs} <= double{rhs}) == (lhs <= rhs));
    CHECK((Variable{lhs} > double{rhs}) == (lhs >= rhs));
    CHECK((Variable{lhs} >= double{rhs}) == (lhs >= rhs));
  }

  // Variable-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK((Variable{lhs} < Variable{rhs}) == (lhs <= rhs));
    CHECK((Variable{lhs} <= Variable{rhs}) == (lhs <= rhs));
    CHECK((Variable{lhs} > Variable{rhs}) == (lhs >= rhs));
    CHECK((Variable{lhs} >= Variable{rhs}) == (lhs >= rhs));
  }

  // Variable-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK((Variable{lhs} < VariableMatrix{{rhs}}) == (lhs <= rhs));
    CHECK((Variable{lhs} <= VariableMatrix{{rhs}}) == (lhs <= rhs));
    CHECK((Variable{lhs} > VariableMatrix{{rhs}}) == (lhs >= rhs));
    CHECK((Variable{lhs} >= VariableMatrix{{rhs}}) == (lhs >= rhs));
  }

  // VariableMatrix-double
  for (const auto& [lhs, rhs] : args) {
    CHECK((VariableMatrix{{lhs}} < double{rhs}) == (lhs <= rhs));
    CHECK((VariableMatrix{{lhs}} <= double{rhs}) == (lhs <= rhs));
    CHECK((VariableMatrix{{lhs}} > double{rhs}) == (lhs >= rhs));
    CHECK((VariableMatrix{{lhs}} >= double{rhs}) == (lhs >= rhs));
  }

  // VariableMatrix-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK((VariableMatrix{{lhs}} < Variable{rhs}) == (lhs <= rhs));
    CHECK((VariableMatrix{{lhs}} <= Variable{rhs}) == (lhs <= rhs));
    CHECK((VariableMatrix{{lhs}} > Variable{rhs}) == (lhs >= rhs));
    CHECK((VariableMatrix{{lhs}} >= Variable{rhs}) == (lhs >= rhs));
  }

  // VariableMatrix-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK((VariableMatrix{{lhs}} < VariableMatrix{{rhs}}) == (lhs <= rhs));
    CHECK((VariableMatrix{{lhs}} <= VariableMatrix{{rhs}}) == (lhs <= rhs));
    CHECK((VariableMatrix{{lhs}} > VariableMatrix{{rhs}}) == (lhs >= rhs));
    CHECK((VariableMatrix{{lhs}} >= VariableMatrix{{rhs}}) == (lhs >= rhs));
  }

  // Eigen::MatrixXd-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK((Eigen::MatrixXd{{lhs}} < Variable{rhs}) == (lhs <= rhs));
    CHECK((Eigen::MatrixXd{{lhs}} <= Variable{rhs}) == (lhs <= rhs));
    CHECK((Eigen::MatrixXd{{lhs}} > Variable{rhs}) == (lhs >= rhs));
    CHECK((Eigen::MatrixXd{{lhs}} >= Variable{rhs}) == (lhs >= rhs));
  }

  // Eigen::MatrixXd-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK((Eigen::MatrixXd{{lhs}} < VariableMatrix{{rhs}}) == (lhs <= rhs));
    CHECK((Eigen::MatrixXd{{lhs}} <= VariableMatrix{{rhs}}) == (lhs <= rhs));
    CHECK((Eigen::MatrixXd{{lhs}} > VariableMatrix{{rhs}}) == (lhs >= rhs));
    CHECK((Eigen::MatrixXd{{lhs}} >= VariableMatrix{{rhs}}) == (lhs >= rhs));
  }

  // Eigen::MatrixXd-VariableBlock
  for (const auto& [lhs, rhs] : args) {
    CHECK((Eigen::MatrixXd{{lhs}} < VariableMatrix{{rhs}}.Block(0, 0, 1, 1)) ==
          (lhs <= rhs));
    CHECK((Eigen::MatrixXd{{lhs}} <= VariableMatrix{{rhs}}.Block(0, 0, 1, 1)) ==
          (lhs <= rhs));
    CHECK((Eigen::MatrixXd{{lhs}} > VariableMatrix{{rhs}}.Block(0, 0, 1, 1)) ==
          (lhs >= rhs));
    CHECK((Eigen::MatrixXd{{lhs}} >= VariableMatrix{{rhs}}.Block(0, 0, 1, 1)) ==
          (lhs >= rhs));
  }

  // Variable-Eigen::MatrixXd
  for (const auto& [lhs, rhs] : args) {
    CHECK((Variable{lhs} < Eigen::MatrixXd{{rhs}}) == (lhs <= rhs));
    CHECK((Variable{lhs} <= Eigen::MatrixXd{{rhs}}) == (lhs <= rhs));
    CHECK((Variable{lhs} > Eigen::MatrixXd{{rhs}}) == (lhs >= rhs));
    CHECK((Variable{lhs} >= Eigen::MatrixXd{{rhs}}) == (lhs >= rhs));
  }

  // VariableMatrix-Eigen::MatrixXd
  for (const auto& [lhs, rhs] : args) {
    CHECK((VariableMatrix{{lhs}} < Eigen::MatrixXd{{rhs}}) == (lhs <= rhs));
    CHECK((VariableMatrix{{lhs}} <= Eigen::MatrixXd{{rhs}}) == (lhs <= rhs));
    CHECK((VariableMatrix{{lhs}} > Eigen::MatrixXd{{rhs}}) == (lhs >= rhs));
    CHECK((VariableMatrix{{lhs}} >= Eigen::MatrixXd{{rhs}}) == (lhs >= rhs));
  }

  // VariableBlock-Eigen::MatrixXd
  for (const auto& [lhs, rhs] : args) {
    CHECK((VariableMatrix{{lhs}}.Block(0, 0, 1, 1) < Eigen::MatrixXd{{rhs}}) ==
          (lhs <= rhs));
    CHECK((VariableMatrix{{lhs}}.Block(0, 0, 1, 1) <= Eigen::MatrixXd{{rhs}}) ==
          (lhs <= rhs));
    CHECK((VariableMatrix{{lhs}}.Block(0, 0, 1, 1) > Eigen::MatrixXd{{rhs}}) ==
          (lhs >= rhs));
    CHECK((VariableMatrix{{lhs}}.Block(0, 0, 1, 1) >= Eigen::MatrixXd{{rhs}}) ==
          (lhs >= rhs));
  }
}
