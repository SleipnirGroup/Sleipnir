// Copyright (c) Sleipnir contributors

#include <array>
#include <tuple>

#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>

#include "catch_string_converters.hpp"

TEST_CASE("constraints - Equality constraint boolean comparison",
          "[constraints]") {
  using slp::Variable;
  using slp::VariableMatrix;

  constexpr std::array args{std::tuple{1.0, 1.0}, std::tuple{1.0, 2.0},
                            std::tuple{2.0, 1.0}};

  // double-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{double{lhs} == Variable{rhs}} == (lhs == rhs));
  }

  // double-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{double{lhs} == VariableMatrix{{rhs}}} == (lhs == rhs));
  }

  // Variable-double
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Variable{lhs} == double{rhs}} == (lhs == rhs));
  }

  // Variable-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Variable{lhs} == Variable{rhs}} == (lhs == rhs));
  }

  // Variable-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Variable{lhs} == VariableMatrix{{rhs}}} == (lhs == rhs));
  }

  // VariableMatrix-double
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{VariableMatrix{{lhs}} == double{rhs}} == (lhs == rhs));
  }

  // VariableMatrix-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{VariableMatrix{{lhs}} == Variable{rhs}} == (lhs == rhs));
  }

  // VariableMatrix-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{VariableMatrix{{lhs}} == VariableMatrix{{rhs}}} == (lhs == rhs));
  }

  // Eigen::MatrixXd-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Eigen::MatrixXd{{lhs}} == Variable{rhs}} == (lhs == rhs));
  }

  // Eigen::MatrixXd-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Eigen::MatrixXd{{lhs}} == VariableMatrix{{rhs}}} ==
          (lhs == rhs));
  }

  // Eigen::MatrixXd-VariableBlock
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Eigen::MatrixXd{{lhs}} ==
               VariableMatrix{{rhs}}.block(0, 0, 1, 1)} == (lhs == rhs));
  }

  // Variable-Eigen::MatrixXd
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Variable{lhs} == Eigen::MatrixXd{{rhs}}} == (lhs == rhs));
  }

  // VariableMatrix-Eigen::MatrixXd
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{VariableMatrix{{lhs}} == Eigen::MatrixXd{{rhs}}} ==
          (lhs == rhs));
  }

  // VariableBlock-Eigen::MatrixXd
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{VariableMatrix{{lhs}}.block(0, 0, 1, 1) ==
               Eigen::MatrixXd{{rhs}}} == (lhs == rhs));
  }
}

// For the purposes of optimization, a < constraint is treated the same as a <=
// constraint
TEST_CASE("constraints - Inequality constraint boolean comparisons",
          "[constraints]") {
  using slp::Variable;
  using slp::VariableMatrix;

  constexpr std::array args{std::tuple{1.0, 1.0}, std::tuple{1.0, 2.0},
                            std::tuple{2.0, 1.0}};

  // double-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{double{lhs} < Variable{rhs}} == (lhs <= rhs));
    CHECK(bool{double{lhs} <= Variable{rhs}} == (lhs <= rhs));
    CHECK(bool{double{lhs} > Variable{rhs}} == (lhs >= rhs));
    CHECK(bool{double{lhs} >= Variable{rhs}} == (lhs >= rhs));
  }

  // double-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{double{lhs} < VariableMatrix{{rhs}}} == (lhs <= rhs));
    CHECK(bool{double{lhs} <= VariableMatrix{{rhs}}} == (lhs <= rhs));
    CHECK(bool{double{lhs} > VariableMatrix{{rhs}}} == (lhs >= rhs));
    CHECK(bool{double{lhs} >= VariableMatrix{{rhs}}} == (lhs >= rhs));
  }

  // Variable-double
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Variable{lhs} < double{rhs}} == (lhs <= rhs));
    CHECK(bool{Variable{lhs} <= double{rhs}} == (lhs <= rhs));
    CHECK(bool{Variable{lhs} > double{rhs}} == (lhs >= rhs));
    CHECK(bool{Variable{lhs} >= double{rhs}} == (lhs >= rhs));
  }

  // Variable-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Variable{lhs} < Variable{rhs}} == (lhs <= rhs));
    CHECK(bool{Variable{lhs} <= Variable{rhs}} == (lhs <= rhs));
    CHECK(bool{Variable{lhs} > Variable{rhs}} == (lhs >= rhs));
    CHECK(bool{Variable{lhs} >= Variable{rhs}} == (lhs >= rhs));
  }

  // Variable-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Variable{lhs} < VariableMatrix{{rhs}}} == (lhs <= rhs));
    CHECK(bool{Variable{lhs} <= VariableMatrix{{rhs}}} == (lhs <= rhs));
    CHECK(bool{Variable{lhs} > VariableMatrix{{rhs}}} == (lhs >= rhs));
    CHECK(bool{Variable{lhs} >= VariableMatrix{{rhs}}} == (lhs >= rhs));
  }

  // VariableMatrix-double
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{VariableMatrix{{lhs}} < double{rhs}} == (lhs <= rhs));
    CHECK(bool{VariableMatrix{{lhs}} <= double{rhs}} == (lhs <= rhs));
    CHECK(bool{VariableMatrix{{lhs}} > double{rhs}} == (lhs >= rhs));
    CHECK(bool{VariableMatrix{{lhs}} >= double{rhs}} == (lhs >= rhs));
  }

  // VariableMatrix-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{VariableMatrix{{lhs}} < Variable{rhs}} == (lhs <= rhs));
    CHECK(bool{VariableMatrix{{lhs}} <= Variable{rhs}} == (lhs <= rhs));
    CHECK(bool{VariableMatrix{{lhs}} > Variable{rhs}} == (lhs >= rhs));
    CHECK(bool{VariableMatrix{{lhs}} >= Variable{rhs}} == (lhs >= rhs));
  }

  // VariableMatrix-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{VariableMatrix{{lhs}} < VariableMatrix{{rhs}}} == (lhs <= rhs));
    CHECK(bool{VariableMatrix{{lhs}} <= VariableMatrix{{rhs}}} == (lhs <= rhs));
    CHECK(bool{VariableMatrix{{lhs}} > VariableMatrix{{rhs}}} == (lhs >= rhs));
    CHECK(bool{VariableMatrix{{lhs}} >= VariableMatrix{{rhs}}} == (lhs >= rhs));
  }

  // Eigen::MatrixXd-Variable
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Eigen::MatrixXd{{lhs}} < Variable{rhs}} == (lhs <= rhs));
    CHECK(bool{Eigen::MatrixXd{{lhs}} <= Variable{rhs}} == (lhs <= rhs));
    CHECK(bool{Eigen::MatrixXd{{lhs}} > Variable{rhs}} == (lhs >= rhs));
    CHECK(bool{Eigen::MatrixXd{{lhs}} >= Variable{rhs}} == (lhs >= rhs));
  }

  // Eigen::MatrixXd-VariableMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Eigen::MatrixXd{{lhs}} < VariableMatrix{{rhs}}} == (lhs <= rhs));
    CHECK(bool{Eigen::MatrixXd{{lhs}} <= VariableMatrix{{rhs}}} ==
          (lhs <= rhs));
    CHECK(bool{Eigen::MatrixXd{{lhs}} > VariableMatrix{{rhs}}} == (lhs >= rhs));
    CHECK(bool{Eigen::MatrixXd{{lhs}} >= VariableMatrix{{rhs}}} ==
          (lhs >= rhs));
  }

  // Eigen::MatrixXd-VariableBlock
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Eigen::MatrixXd{{lhs}} <
               VariableMatrix{{rhs}}.block(0, 0, 1, 1)} == (lhs <= rhs));
    CHECK(bool{Eigen::MatrixXd{{lhs}} <=
               VariableMatrix{{rhs}}.block(0, 0, 1, 1)} == (lhs <= rhs));
    CHECK(bool{Eigen::MatrixXd{{lhs}} >
               VariableMatrix{{rhs}}.block(0, 0, 1, 1)} == (lhs >= rhs));
    CHECK(bool{Eigen::MatrixXd{{lhs}} >=
               VariableMatrix{{rhs}}.block(0, 0, 1, 1)} == (lhs >= rhs));
  }

  // Variable-Eigen::MatrixXd
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Variable{lhs} < Eigen::MatrixXd{{rhs}}} == (lhs <= rhs));
    CHECK(bool{Variable{lhs} <= Eigen::MatrixXd{{rhs}}} == (lhs <= rhs));
    CHECK(bool{Variable{lhs} > Eigen::MatrixXd{{rhs}}} == (lhs >= rhs));
    CHECK(bool{Variable{lhs} >= Eigen::MatrixXd{{rhs}}} == (lhs >= rhs));
  }

  // VariableMatrix-Eigen::MatrixXd
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{VariableMatrix{{lhs}} < Eigen::MatrixXd{{rhs}}} == (lhs <= rhs));
    CHECK(bool{VariableMatrix{{lhs}} <= Eigen::MatrixXd{{rhs}}} ==
          (lhs <= rhs));
    CHECK(bool{VariableMatrix{{lhs}} > Eigen::MatrixXd{{rhs}}} == (lhs >= rhs));
    CHECK(bool{VariableMatrix{{lhs}} >= Eigen::MatrixXd{{rhs}}} ==
          (lhs >= rhs));
  }

  // VariableBlock-Eigen::MatrixXd
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{VariableMatrix{{lhs}}.block(0, 0, 1, 1) <
               Eigen::MatrixXd{{rhs}}} == (lhs <= rhs));
    CHECK(bool{VariableMatrix{{lhs}}.block(0, 0, 1, 1) <=
               Eigen::MatrixXd{{rhs}}} == (lhs <= rhs));
    CHECK(bool{VariableMatrix{{lhs}}.block(0, 0, 1, 1) >
               Eigen::MatrixXd{{rhs}}} == (lhs >= rhs));
    CHECK(bool{VariableMatrix{{lhs}}.block(0, 0, 1, 1) >=
               Eigen::MatrixXd{{rhs}}} == (lhs >= rhs));
  }
}

TEST_CASE("constraints - Equality constraint concatenation", "[constraints]") {
  using slp::EqualityConstraints;
  using slp::Variable;

  EqualityConstraints eq1 = Variable{1.0} == Variable{1.0};
  EqualityConstraints eq2 = Variable{1.0} == Variable{2.0};
  EqualityConstraints eqs{eq1, eq2};

  CHECK(eq1.constraints.size() == 1);
  CHECK(eq2.constraints.size() == 1);
  CHECK(eqs.constraints.size() == 2);

  CHECK(eqs.constraints[0].value() == eq1.constraints[0].value());
  CHECK(eqs.constraints[1].value() == eq2.constraints[0].value());

  CHECK(eq1);
  CHECK_FALSE(eq2);
  CHECK_FALSE(eqs);
}

TEST_CASE("constraints - Inequality constraint concatenation",
          "[constraints]") {
  using slp::InequalityConstraints;
  using slp::Variable;

  InequalityConstraints ineq1 = Variable{2.0} < Variable{1.0};
  InequalityConstraints ineq2 = Variable{1.0} < Variable{2.0};
  InequalityConstraints ineqs{ineq1, ineq2};

  CHECK(ineq1.constraints.size() == 1);
  CHECK(ineq2.constraints.size() == 1);
  CHECK(ineqs.constraints.size() == 2);

  CHECK(ineqs.constraints[0].value() == ineq1.constraints[0].value());
  CHECK(ineqs.constraints[1].value() == ineq2.constraints[0].value());

  CHECK_FALSE(ineq1);
  CHECK(ineq2);
  CHECK_FALSE(ineqs);
}
