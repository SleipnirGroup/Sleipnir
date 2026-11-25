// Copyright (c) Sleipnir contributors

#include <array>
#include <tuple>

#include <Eigen/Core>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/real.hpp>
#include <sleipnir/autodiff/real_matrix.hpp>

#include "catch_string_converters.hpp"
#include "scalar_types.hpp"

TEMPLATE_TEST_CASE("constraints - Equality constraint boolean comparison",
                   "[constraints]", SCALAR_TYPES) {
  using T = TestType;

  constexpr std::array args{std::tuple{T(1), T(1)}, std::tuple{T(1), T(2)},
                            std::tuple{T(2), T(1)}};

  // T-Real
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{T{lhs} == slp::Real<T>{rhs}} == (lhs == rhs));
  }

  // T-RealMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{T{lhs} == slp::RealMatrix<T>{{rhs}}} == (lhs == rhs));
  }

  // Real-T
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{slp::Real<T>{lhs} == T{rhs}} == (lhs == rhs));
  }

  // Real-Real
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{slp::Real<T>{lhs} == slp::Real<T>{rhs}} == (lhs == rhs));
  }

  // Real-RealMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{slp::Real<T>{lhs} == slp::RealMatrix<T>{{rhs}}} == (lhs == rhs));
  }

  // RealMatrix-T
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{slp::RealMatrix<T>{{lhs}} == T{rhs}} == (lhs == rhs));
  }

  // RealMatrix-Real
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{slp::RealMatrix<T>{{lhs}} == slp::Real<T>{rhs}} == (lhs == rhs));
  }

  // RealMatrix-RealMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{slp::RealMatrix<T>{{lhs}} == slp::RealMatrix<T>{{rhs}}} ==
          (lhs == rhs));
  }

  // Eigen::MatrixX<T>-Real
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Eigen::MatrixX<T>{{lhs}} == slp::Real<T>{rhs}} == (lhs == rhs));
  }

  // Eigen::MatrixX<T>-RealMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Eigen::MatrixX<T>{{lhs}} == slp::RealMatrix<T>{{rhs}}} ==
          (lhs == rhs));
  }

  // Eigen::MatrixX<T>-RealSubmatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Eigen::MatrixX<T>{{lhs}} ==
               slp::RealMatrix<T>{{rhs}}.block(0, 0, 1, 1)} == (lhs == rhs));
  }

  // Real-Eigen::MatrixX<T>
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{slp::Real<T>{lhs} == Eigen::MatrixX<T>{{rhs}}} == (lhs == rhs));
  }

  // RealMatrix-Eigen::MatrixX<T>
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{slp::RealMatrix<T>{{lhs}} == Eigen::MatrixX<T>{{rhs}}} ==
          (lhs == rhs));
  }

  // RealSubmatrix-Eigen::MatrixX<T>
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{slp::RealMatrix<T>{{lhs}}.block(0, 0, 1, 1) ==
               Eigen::MatrixX<T>{{rhs}}} == (lhs == rhs));
  }
}

// The solver treats a < constraint the same as a <= constraint, but the
// boolean conversion distinguishes them
TEMPLATE_TEST_CASE("constraints - Inequality constraint boolean comparisons",
                   "[constraints]", SCALAR_TYPES) {
  using T = TestType;

  constexpr std::array args{std::tuple{T(1), T(1)}, std::tuple{T(1), T(2)},
                            std::tuple{T(2), T(1)}};

  // T-Real
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{T{lhs} < slp::Real<T>{rhs}} == (lhs < rhs));
    CHECK(bool{T{lhs} <= slp::Real<T>{rhs}} == (lhs <= rhs));
    CHECK(bool{T{lhs} > slp::Real<T>{rhs}} == (lhs > rhs));
    CHECK(bool{T{lhs} >= slp::Real<T>{rhs}} == (lhs >= rhs));
  }

  // T-RealMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{T{lhs} < slp::RealMatrix<T>{{rhs}}} == (lhs < rhs));
    CHECK(bool{T{lhs} <= slp::RealMatrix<T>{{rhs}}} == (lhs <= rhs));
    CHECK(bool{T{lhs} > slp::RealMatrix<T>{{rhs}}} == (lhs > rhs));
    CHECK(bool{T{lhs} >= slp::RealMatrix<T>{{rhs}}} == (lhs >= rhs));
  }

  // Real-T
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{slp::Real<T>{lhs} < T{rhs}} == (lhs < rhs));
    CHECK(bool{slp::Real<T>{lhs} <= T{rhs}} == (lhs <= rhs));
    CHECK(bool{slp::Real<T>{lhs} > T{rhs}} == (lhs > rhs));
    CHECK(bool{slp::Real<T>{lhs} >= T{rhs}} == (lhs >= rhs));
  }

  // Real-Real
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{slp::Real<T>{lhs} < slp::Real<T>{rhs}} == (lhs < rhs));
    CHECK(bool{slp::Real<T>{lhs} <= slp::Real<T>{rhs}} == (lhs <= rhs));
    CHECK(bool{slp::Real<T>{lhs} > slp::Real<T>{rhs}} == (lhs > rhs));
    CHECK(bool{slp::Real<T>{lhs} >= slp::Real<T>{rhs}} == (lhs >= rhs));
  }

  // Real-RealMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{slp::Real<T>{lhs} < slp::RealMatrix<T>{{rhs}}} == (lhs < rhs));
    CHECK(bool{slp::Real<T>{lhs} <= slp::RealMatrix<T>{{rhs}}} == (lhs <= rhs));
    CHECK(bool{slp::Real<T>{lhs} > slp::RealMatrix<T>{{rhs}}} == (lhs > rhs));
    CHECK(bool{slp::Real<T>{lhs} >= slp::RealMatrix<T>{{rhs}}} == (lhs >= rhs));
  }

  // RealMatrix-T
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{slp::RealMatrix<T>{{lhs}} < T{rhs}} == (lhs < rhs));
    CHECK(bool{slp::RealMatrix<T>{{lhs}} <= T{rhs}} == (lhs <= rhs));
    CHECK(bool{slp::RealMatrix<T>{{lhs}} > T{rhs}} == (lhs > rhs));
    CHECK(bool{slp::RealMatrix<T>{{lhs}} >= T{rhs}} == (lhs >= rhs));
  }

  // RealMatrix-Real
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{slp::RealMatrix<T>{{lhs}} < slp::Real<T>{rhs}} == (lhs < rhs));
    CHECK(bool{slp::RealMatrix<T>{{lhs}} <= slp::Real<T>{rhs}} == (lhs <= rhs));
    CHECK(bool{slp::RealMatrix<T>{{lhs}} > slp::Real<T>{rhs}} == (lhs > rhs));
    CHECK(bool{slp::RealMatrix<T>{{lhs}} >= slp::Real<T>{rhs}} == (lhs >= rhs));
  }

  // RealMatrix-RealMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{slp::RealMatrix<T>{{lhs}} < slp::RealMatrix<T>{{rhs}}} ==
          (lhs < rhs));
    CHECK(bool{slp::RealMatrix<T>{{lhs}} <= slp::RealMatrix<T>{{rhs}}} ==
          (lhs <= rhs));
    CHECK(bool{slp::RealMatrix<T>{{lhs}} > slp::RealMatrix<T>{{rhs}}} ==
          (lhs > rhs));
    CHECK(bool{slp::RealMatrix<T>{{lhs}} >= slp::RealMatrix<T>{{rhs}}} ==
          (lhs >= rhs));
  }

  // Eigen::MatrixX<T>-Real
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Eigen::MatrixX<T>{{lhs}} < slp::Real<T>{rhs}} == (lhs < rhs));
    CHECK(bool{Eigen::MatrixX<T>{{lhs}} <= slp::Real<T>{rhs}} == (lhs <= rhs));
    CHECK(bool{Eigen::MatrixX<T>{{lhs}} > slp::Real<T>{rhs}} == (lhs > rhs));
    CHECK(bool{Eigen::MatrixX<T>{{lhs}} >= slp::Real<T>{rhs}} == (lhs >= rhs));
  }

  // Eigen::MatrixX<T>-RealMatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Eigen::MatrixX<T>{{lhs}} < slp::RealMatrix<T>{{rhs}}} ==
          (lhs < rhs));
    CHECK(bool{Eigen::MatrixX<T>{{lhs}} <= slp::RealMatrix<T>{{rhs}}} ==
          (lhs <= rhs));
    CHECK(bool{Eigen::MatrixX<T>{{lhs}} > slp::RealMatrix<T>{{rhs}}} ==
          (lhs > rhs));
    CHECK(bool{Eigen::MatrixX<T>{{lhs}} >= slp::RealMatrix<T>{{rhs}}} ==
          (lhs >= rhs));
  }

  // Eigen::MatrixX<T>-RealSubmatrix
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{Eigen::MatrixX<T>{{lhs}} <
               slp::RealMatrix<T>{{rhs}}.block(0, 0, 1, 1)} == (lhs < rhs));
    CHECK(bool{Eigen::MatrixX<T>{{lhs}} <=
               slp::RealMatrix<T>{{rhs}}.block(0, 0, 1, 1)} == (lhs <= rhs));
    CHECK(bool{Eigen::MatrixX<T>{{lhs}} >
               slp::RealMatrix<T>{{rhs}}.block(0, 0, 1, 1)} == (lhs > rhs));
    CHECK(bool{Eigen::MatrixX<T>{{lhs}} >=
               slp::RealMatrix<T>{{rhs}}.block(0, 0, 1, 1)} == (lhs >= rhs));
  }

  // Real-Eigen::MatrixX<T>
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{slp::Real<T>{lhs} < Eigen::MatrixX<T>{{rhs}}} == (lhs < rhs));
    CHECK(bool{slp::Real<T>{lhs} <= Eigen::MatrixX<T>{{rhs}}} == (lhs <= rhs));
    CHECK(bool{slp::Real<T>{lhs} > Eigen::MatrixX<T>{{rhs}}} == (lhs > rhs));
    CHECK(bool{slp::Real<T>{lhs} >= Eigen::MatrixX<T>{{rhs}}} == (lhs >= rhs));
  }

  // RealMatrix-Eigen::MatrixX<T>
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{slp::RealMatrix<T>{{lhs}} < Eigen::MatrixX<T>{{rhs}}} ==
          (lhs < rhs));
    CHECK(bool{slp::RealMatrix<T>{{lhs}} <= Eigen::MatrixX<T>{{rhs}}} ==
          (lhs <= rhs));
    CHECK(bool{slp::RealMatrix<T>{{lhs}} > Eigen::MatrixX<T>{{rhs}}} ==
          (lhs > rhs));
    CHECK(bool{slp::RealMatrix<T>{{lhs}} >= Eigen::MatrixX<T>{{rhs}}} ==
          (lhs >= rhs));
  }

  // RealSubmatrix-Eigen::MatrixX<T>
  for (const auto& [lhs, rhs] : args) {
    CHECK(bool{slp::RealMatrix<T>{{lhs}}.block(0, 0, 1, 1) <
               Eigen::MatrixX<T>{{rhs}}} == (lhs < rhs));
    CHECK(bool{slp::RealMatrix<T>{{lhs}}.block(0, 0, 1, 1) <=
               Eigen::MatrixX<T>{{rhs}}} == (lhs <= rhs));
    CHECK(bool{slp::RealMatrix<T>{{lhs}}.block(0, 0, 1, 1) >
               Eigen::MatrixX<T>{{rhs}}} == (lhs > rhs));
    CHECK(bool{slp::RealMatrix<T>{{lhs}}.block(0, 0, 1, 1) >=
               Eigen::MatrixX<T>{{rhs}}} == (lhs >= rhs));
  }
}

TEMPLATE_TEST_CASE("constraints - Equality constraint concatenation",
                   "[constraints]", SCALAR_TYPES) {
  using T = TestType;

  slp::EqualityConstraints eq1 = slp::Real<T>{1} == slp::Real<T>{1};
  slp::EqualityConstraints eq2 = slp::Real<T>{1} == slp::Real<T>{2};
  slp::EqualityConstraints eqs{eq1, eq2};

  CHECK(eq1.constraints.size() == 1);
  CHECK(eq2.constraints.size() == 1);
  CHECK(eqs.constraints.size() == 2);

  CHECK(eqs.constraints[0].value() == eq1.constraints[0].value());
  CHECK(eqs.constraints[1].value() == eq2.constraints[0].value());

  CHECK(bool{eq1});
  CHECK_FALSE(bool{eq2});
  CHECK_FALSE(bool{eqs});
}

TEMPLATE_TEST_CASE("constraints - Inequality constraint concatenation",
                   "[constraints]", SCALAR_TYPES) {
  using T = TestType;

  slp::InequalityConstraints ineq1 = slp::Real<T>{2} < slp::Real<T>{1};
  slp::InequalityConstraints ineq2 = slp::Real<T>{1} < slp::Real<T>{2};
  slp::InequalityConstraints ineqs{ineq1, ineq2};

  CHECK(ineq1.constraints.size() == 1);
  CHECK(ineq2.constraints.size() == 1);
  CHECK(ineqs.constraints.size() == 2);

  CHECK(ineqs.constraints[0].value() == ineq1.constraints[0].value());
  CHECK(ineqs.constraints[1].value() == ineq2.constraints[0].value());

  CHECK_FALSE(bool{ineq1});
  CHECK(bool{ineq2});
  CHECK_FALSE(bool{ineqs});

  // Each constraint keeps its own strictness
  slp::InequalityConstraints strict = slp::Real<T>{1} < slp::Real<T>{1};
  slp::InequalityConstraints non_strict = slp::Real<T>{1} <= slp::Real<T>{1};
  CHECK_FALSE(bool{slp::InequalityConstraints{strict, non_strict}});
  CHECK_FALSE(bool{slp::InequalityConstraints{non_strict, strict}});
  CHECK(bool{slp::InequalityConstraints{non_strict, non_strict}});
  CHECK(bool{slp::InequalityConstraints{ineq2, non_strict}});
}
