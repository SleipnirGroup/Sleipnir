// Copyright (c) Sleipnir contributors

#include <functional>
#include <iterator>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/VariableMatrix.hpp>
#include <sleipnir/optimization/Constraints.hpp>

TEST_CASE("VariableMatrix - Construct from Eigen::MatrixBase",
          "[VariableMatrix]") {
  sleipnir::VariableMatrix mat{
      Eigen::MatrixXd{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}};

  Eigen::MatrixXd expected{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  CHECK(mat.Value() == expected);
}

TEST_CASE("VariableMatrix - Construct from Eigen::DiagonalBase",
          "[VariableMatrix]") {
  sleipnir::VariableMatrix mat{Eigen::VectorXd{{1.0, 2.0, 3.0}}.asDiagonal()};

  Eigen::MatrixXd expected{{1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 3.0}};
  CHECK(mat.Value() == expected);
}

TEST_CASE("VariableMatrix - Assignment to default", "[VariableMatrix]") {
  sleipnir::VariableMatrix mat;

  CHECK(mat.Rows() == 0);
  CHECK(mat.Cols() == 0);

  mat = sleipnir::VariableMatrix{2, 2};

  CHECK(mat.Rows() == 2);
  CHECK(mat.Cols() == 2);
  CHECK(mat(0, 0) == 0.0);
  CHECK(mat(0, 1) == 0.0);
  CHECK(mat(1, 0) == 0.0);
  CHECK(mat(1, 1) == 0.0);

  mat(0, 0) = 1.0;
  mat(0, 1) = 2.0;
  mat(1, 0) = 3.0;
  mat(1, 1) = 4.0;

  CHECK(mat(0, 0) == 1.0);
  CHECK(mat(0, 1) == 2.0);
  CHECK(mat(1, 0) == 3.0);
  CHECK(mat(1, 1) == 4.0);
}

TEST_CASE("VariableMatrix - Assignment aliasing", "[VariableMatrix]") {
  sleipnir::VariableMatrix A{{1.0, 2.0}, {3.0, 4.0}};
  sleipnir::VariableMatrix B{{5.0, 6.0}, {7.0, 8.0}};

  Eigen::MatrixXd expectedA{{1.0, 2.0}, {3.0, 4.0}};
  Eigen::MatrixXd expectedB{{5.0, 6.0}, {7.0, 8.0}};
  CHECK(A == expectedA);
  CHECK(B == expectedB);

  A = B;

  CHECK(A == expectedB);
  CHECK(B == expectedB);

  B(0, 0).SetValue(2.0);
  expectedB(0, 0) = 2.0;

  CHECK(A == expectedB);
  CHECK(B == expectedB);
}

TEST_CASE("VariableMatrix - Block() member function", "[VariableMatrix]") {
  sleipnir::VariableMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

  // Block assignment
  A.Block(1, 1, 2, 2) = Eigen::Matrix<double, 2, 2>{{10.0, 11.0}, {12.0, 13.0}};

  Eigen::Matrix<double, 3, 3> expected1{
      {1.0, 2.0, 3.0}, {4.0, 10.0, 11.0}, {7.0, 12.0, 13.0}};
  CHECK(A.Value() == expected1);

  // Block-of-block assignment
  A.Block(1, 1, 2, 2).Block(1, 1, 1, 1) = 14.0;

  Eigen::Matrix<double, 3, 3> expected2{
      {1.0, 2.0, 3.0}, {4.0, 10.0, 11.0}, {7.0, 12.0, 14.0}};
  CHECK(A.Value() == expected2);
}

TEST_CASE("VariableMatrix - Iterators", "[VariableMatrix]") {
  sleipnir::VariableMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

  // VariableMatrix iterator
  CHECK(std::distance(A.begin(), A.end()) == 9);

  int i = 1;
  for (auto& elem : A) {
    CHECK(elem.Value() == i);
    ++i;
  }

  // VariableMatrix const_iterator
  CHECK(std::distance(A.cbegin(), A.cend()) == 9);

  // Value() isn't const-qualified
#if 0
  i = 1;
  for (const auto& elem : A) {
    CHECK(elem.Value() == i);
    ++i;
  }
#endif

  auto Asub = A.Block(2, 1, 1, 2);

  // VariableBlock iterator
  CHECK(std::distance(Asub.begin(), Asub.end()) == 2);

  i = 8;
  for (auto& elem : Asub) {
    CHECK(elem.Value() == i);
    ++i;
  }

  // VariableBlock const_iterator
  CHECK(std::distance(Asub.begin(), Asub.end()) == 2);

  // Value() isn't const-qualified
#if 0
  i = 8;
  for (const auto& elem : Asub) {
    CHECK(elem.Value() == i);
    ++i;
  }
#endif
}

TEST_CASE("VariableMatrix - CwiseTransform()", "[VariableMatrix]") {
  // VariableMatrix CwiseTransform
  sleipnir::VariableMatrix A{{-2.0, -3.0, -4.0}, {-5.0, -6.0, -7.0}};

  sleipnir::VariableMatrix result1 = A.CwiseTransform(sleipnir::abs);
  Eigen::Matrix<double, 2, 3> expected1{{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}};

  // Don't modify original matrix
  CHECK(A.Value() == -expected1);

  CHECK(result1.Value() == expected1);

  // VariableBlock CwiseTransform
  auto Asub = A.Block(0, 0, 2, 2);

  sleipnir::VariableMatrix result2 = Asub.CwiseTransform(sleipnir::abs);
  Eigen::Matrix<double, 2, 2> expected2{{2.0, 3.0}, {5.0, 6.0}};

  // Don't modify original matrix
  CHECK(A.Value() == -expected1);
  CHECK(Asub.Value() == -expected2);

  CHECK(result2.Value() == expected2);
}

TEST_CASE("VariableMatrix - Zero() static function", "[VariableMatrix]") {
  auto A = sleipnir::VariableMatrix::Zero(2, 3);

  for (auto& elem : A) {
    CHECK(elem.Value() == 0.0);
  }
}

TEST_CASE("VariableMatrix - Ones() static function", "[VariableMatrix]") {
  auto A = sleipnir::VariableMatrix::Ones(2, 3);

  for (auto& elem : A) {
    CHECK(elem.Value() == 1.0);
  }
}

TEST_CASE("VariableMatrix - CwiseReduce()", "[VariableMatrix]") {
  sleipnir::VariableMatrix A{{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}};
  sleipnir::VariableMatrix B{{8.0, 9.0, 10.0}, {11.0, 12.0, 13.0}};
  sleipnir::VariableMatrix result =
      sleipnir::CwiseReduce(A, B, std::multiplies<>{});

  Eigen::Matrix<double, 2, 3> expected{{16.0, 27.0, 40.0}, {55.0, 72.0, 91.0}};
  CHECK(result.Value() == expected);
}

TEST_CASE("VariableMatrix - Block() free function", "[VariableMatrix]") {
  sleipnir::VariableMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  sleipnir::VariableMatrix B{{7.0}, {8.0}};

  sleipnir::VariableMatrix mat1 = sleipnir::Block({{A, B}});
  Eigen::Matrix<double, 2, 4> expected1{{1.0, 2.0, 3.0, 7.0},
                                        {4.0, 5.0, 6.0, 8.0}};
  CHECK(mat1.Rows() == 2);
  CHECK(mat1.Cols() == 4);
  CHECK(mat1.Value() == expected1);

  sleipnir::VariableMatrix C{{9.0, 10.0, 11.0, 12.0}};

  sleipnir::VariableMatrix mat2 = sleipnir::Block({{A, B}, {C}});
  Eigen::Matrix<double, 3, 4> expected2{
      {1.0, 2.0, 3.0, 7.0}, {4.0, 5.0, 6.0, 8.0}, {9.0, 10.0, 11.0, 12.0}};
  CHECK(mat2.Rows() == 3);
  CHECK(mat2.Cols() == 4);
  CHECK(mat2.Value() == expected2);
}

TEST_CASE("VariableMatrix - Solve() free function", "[VariableMatrix]") {
  sleipnir::VariableMatrix A1{{1.0, 2.0}, {3.0, 4.0}};
  sleipnir::VariableMatrix B1{{5.0}, {6.0}};
  sleipnir::VariableMatrix X1 = sleipnir::Solve(A1, B1);

  Eigen::Matrix<double, 2, 1> expected1{{-4.0}, {4.5}};
  CHECK(X1.Rows() == 2);
  CHECK(X1.Cols() == 1);
  CHECK(A1.Value() * X1.Value() == B1.Value());
  CHECK(X1.Value() == expected1);

  sleipnir::VariableMatrix A2{
      {1.0, 2.0, 3.0}, {-4.0, -5.0, 6.0}, {7.0, 8.0, 9.0}};
  sleipnir::VariableMatrix B2{{10.0}, {11.0}, {12.0}};
  sleipnir::VariableMatrix X2 = sleipnir::Solve(A2, B2);

  Eigen::Matrix<double, 3, 1> expected2{{-7.5}, {6.0}, {11.0 / 6.0}};
  CHECK(X2.Rows() == 3);
  CHECK(X2.Cols() == 1);
  CHECK((A2.Value() * X2.Value() - B2.Value()).norm() < 1e-12);
  CHECK((X2.Value() - expected2).norm() < 1e-12);
}
