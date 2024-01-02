// Copyright (c) Sleipnir contributors

#include <functional>
#include <iterator>

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <sleipnir/autodiff/VariableMatrix.hpp>
#include <sleipnir/optimization/Constraints.hpp>

TEST(VariableMatrixTest, ConstructFromEigenMatrixBase) {
  sleipnir::VariableMatrix mat{
      Eigen::MatrixXd{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}};

  Eigen::MatrixXd expected{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  EXPECT_EQ(expected, mat.Value());
}

TEST(VariableMatrixTest, ConstructFromEigenDiagonalBase) {
  sleipnir::VariableMatrix mat{Eigen::VectorXd{{1.0, 2.0, 3.0}}.asDiagonal()};

  Eigen::MatrixXd expected{{1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 3.0}};
  EXPECT_EQ(expected, mat.Value());
}

TEST(VariableMatrixTest, AssignmentToDefault) {
  sleipnir::VariableMatrix mat;

  EXPECT_EQ(0, mat.Rows());
  EXPECT_EQ(0, mat.Cols());

  mat = sleipnir::VariableMatrix{2, 2};

  EXPECT_EQ(2, mat.Rows());
  EXPECT_EQ(2, mat.Cols());
  EXPECT_EQ(0.0, mat(0, 0));
  EXPECT_EQ(0.0, mat(0, 1));
  EXPECT_EQ(0.0, mat(1, 0));
  EXPECT_EQ(0.0, mat(1, 1));

  mat(0, 0) = 1.0;
  mat(0, 1) = 2.0;
  mat(1, 0) = 3.0;
  mat(1, 1) = 4.0;

  EXPECT_EQ(1.0, mat(0, 0));
  EXPECT_EQ(2.0, mat(0, 1));
  EXPECT_EQ(3.0, mat(1, 0));
  EXPECT_EQ(4.0, mat(1, 1));
}

TEST(VariableMatrixTest, AssignmentAliasing) {
  sleipnir::VariableMatrix A{{1.0, 2.0}, {3.0, 4.0}};
  sleipnir::VariableMatrix B{{5.0, 6.0}, {7.0, 8.0}};

  Eigen::MatrixXd expectedA{{1.0, 2.0}, {3.0, 4.0}};
  Eigen::MatrixXd expectedB{{5.0, 6.0}, {7.0, 8.0}};
  EXPECT_EQ(expectedA, A);
  EXPECT_EQ(expectedB, B);

  A = B;

  EXPECT_EQ(expectedB, A);
  EXPECT_EQ(expectedB, B);

  B(0, 0).SetValue(2.0);
  expectedB(0, 0) = 2.0;

  EXPECT_EQ(expectedB, A);
  EXPECT_EQ(expectedB, B);
}

TEST(VariableMatrixTest, Iterators) {
  sleipnir::VariableMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

  // VariableMatrix iterator
  EXPECT_EQ(9, std::distance(A.begin(), A.end()));

  int i = 1;
  for (auto& elem : A) {
    EXPECT_EQ(i, elem.Value());
    ++i;
  }

  // VariableMatrix const_iterator
  EXPECT_EQ(9, std::distance(A.cbegin(), A.cend()));

  i = 1;
  for (const auto& elem : A) {
    EXPECT_EQ(i, elem.Value());
    ++i;
  }

  auto Asub = A.Block(2, 1, 1, 2);

  // VariableBlock iterator
  EXPECT_EQ(2, std::distance(Asub.begin(), Asub.end()));

  i = 8;
  for (auto& elem : Asub) {
    EXPECT_EQ(i, elem.Value());
    ++i;
  }

  // VariableBlock const_iterator
  EXPECT_EQ(2, std::distance(Asub.begin(), Asub.end()));

  i = 8;
  for (const auto& elem : Asub) {
    EXPECT_EQ(i, elem.Value());
    ++i;
  }
}

TEST(VariableMatrixTest, CwiseTransform) {
  // VariableMatrix CwiseTransform
  sleipnir::VariableMatrix A{{-2.0, -3.0, -4.0}, {-5.0, -6.0, -7.0}};

  sleipnir::VariableMatrix result1 = A.CwiseTransform(sleipnir::abs);
  Eigen::Matrix<double, 2, 3> expected1{{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}};

  // Don't modify original matrix
  EXPECT_EQ(-expected1, A.Value());

  EXPECT_EQ(expected1, result1.Value());

  // VariableBlock CwiseTransform
  auto Asub = A.Block(0, 0, 2, 2);

  sleipnir::VariableMatrix result2 = Asub.CwiseTransform(sleipnir::abs);
  Eigen::Matrix<double, 2, 2> expected2{{2.0, 3.0}, {5.0, 6.0}};

  // Don't modify original matrix
  EXPECT_EQ(-expected1, A.Value());
  EXPECT_EQ(-expected2, Asub.Value());

  EXPECT_EQ(expected2, result2.Value());
}

TEST(VariableMatrixTest, ZeroStaticFunction) {
  auto A = sleipnir::VariableMatrix::Zero(2, 3);

  for (const auto& elem : A) {
    EXPECT_EQ(0.0, elem.Value());
  }
}

TEST(VariableMatrixTest, OnesStaticFunction) {
  auto A = sleipnir::VariableMatrix::Ones(2, 3);

  for (const auto& elem : A) {
    EXPECT_EQ(1.0, elem.Value());
  }
}

TEST(VariableMatrixTest, CwiseReduce) {
  sleipnir::VariableMatrix A{{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}};
  sleipnir::VariableMatrix B{{8.0, 9.0, 10.0}, {11.0, 12.0, 13.0}};
  sleipnir::VariableMatrix result =
      sleipnir::CwiseReduce(A, B, std::multiplies<>{});

  Eigen::Matrix<double, 2, 3> expected{{16.0, 27.0, 40.0}, {55.0, 72.0, 91.0}};
  EXPECT_EQ(expected, result.Value());
}

TEST(VariableMatrixTest, BlockFreeFunction) {
  sleipnir::VariableMatrix A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  sleipnir::VariableMatrix B{{7.0}, {8.0}};

  sleipnir::VariableMatrix mat1 = sleipnir::Block({{A, B}});
  Eigen::Matrix<double, 2, 4> expected1{{1.0, 2.0, 3.0, 7.0},
                                        {4.0, 5.0, 6.0, 8.0}};
  EXPECT_EQ(2, mat1.Rows());
  EXPECT_EQ(4, mat1.Cols());
  EXPECT_EQ(expected1, mat1.Value());

  sleipnir::VariableMatrix C{{9.0, 10.0, 11.0, 12.0}};

  sleipnir::VariableMatrix mat2 = sleipnir::Block({{A, B}, {C}});
  Eigen::Matrix<double, 3, 4> expected2{
      {1.0, 2.0, 3.0, 7.0}, {4.0, 5.0, 6.0, 8.0}, {9.0, 10.0, 11.0, 12.0}};
  EXPECT_EQ(3, mat2.Rows());
  EXPECT_EQ(4, mat2.Cols());
  EXPECT_EQ(expected2, mat2.Value());
}
