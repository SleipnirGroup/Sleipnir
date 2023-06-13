// Copyright (c) Sleipnir contributors

#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include <sleipnir/util/SparseUtil.hpp>

TEST(SparseUtilTest, SparseDiagonal) {
  Eigen::Vector<double, 5> vec{-2.0, -1.0, 0.0, 1.0, 2.0};

  Eigen::Matrix<double, 5, 5> expected{{-2.0, 0.0, 0.0, 0.0, 0.0},
                                       {0.0, -1.0, 0.0, 0.0, 0.0},
                                       {0.0, 0.0, 0.0, 0.0, 0.0},
                                       {0.0, 0.0, 0.0, 1.0, 0.0},
                                       {0.0, 0.0, 0.0, 0.0, 2.0}};
  Eigen::SparseMatrix<double> actual = sleipnir::SparseDiagonal(vec);
  EXPECT_EQ(5, actual.rows());
  EXPECT_EQ(5, actual.cols());
  EXPECT_EQ(5, actual.nonZeros());

  for (int row = 0; row < expected.rows(); ++row) {
    for (int col = 0; col < expected.cols(); ++col) {
      EXPECT_DOUBLE_EQ(expected(row, col), actual.coeff(row, col));
    }
  }
}

TEST(SparseUtilTest, SparseIdentitySquare) {
  Eigen::SparseMatrix<double> actual = sleipnir::SparseIdentity(5, 5);
  EXPECT_EQ(5, actual.rows());
  EXPECT_EQ(5, actual.cols());
  EXPECT_EQ(5, actual.nonZeros());

  for (int row = 0; row < 5; ++row) {
    for (int col = 0; col < 5; ++col) {
      if (row == col) {
        EXPECT_DOUBLE_EQ(1.0, actual.coeff(row, col));
      } else {
        EXPECT_DOUBLE_EQ(0.0, actual.coeff(row, col));
      }
    }
  }
}

TEST(SparseUtilTest, SparseIdentityNonsquareTall) {
  Eigen::SparseMatrix<double> actual = sleipnir::SparseIdentity(3, 2);
  EXPECT_EQ(3, actual.rows());
  EXPECT_EQ(2, actual.cols());
  EXPECT_EQ(2, actual.nonZeros());

  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 2; ++col) {
      if (row == col) {
        EXPECT_DOUBLE_EQ(1.0, actual.coeff(row, col));
      } else {
        EXPECT_DOUBLE_EQ(0.0, actual.coeff(row, col));
      }
    }
  }
}

TEST(SparseUtilTest, SparseIdentityNonsquareWide) {
  Eigen::SparseMatrix<double> actual = sleipnir::SparseIdentity(2, 3);
  EXPECT_EQ(2, actual.rows());
  EXPECT_EQ(3, actual.cols());
  EXPECT_EQ(2, actual.nonZeros());

  for (int row = 0; row < 2; ++row) {
    for (int col = 0; col < 3; ++col) {
      if (row == col) {
        EXPECT_DOUBLE_EQ(1.0, actual.coeff(row, col));
      } else {
        EXPECT_DOUBLE_EQ(0.0, actual.coeff(row, col));
      }
    }
  }
}

TEST(SparseUtilTest, SparseLpNorm) {
  // Largest value being negative ensures absolute value is taken in ∞-norm
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.emplace_back(0, 0, -5.0);
  triplets.emplace_back(0, 1, -4.0);
  triplets.emplace_back(0, 2, -3.0);
  triplets.emplace_back(1, 0, -2.0);
  triplets.emplace_back(1, 1, -1.0);
  triplets.emplace_back(1, 2, 0.0);
  triplets.emplace_back(2, 0, 1.0);
  triplets.emplace_back(2, 1, 2.0);
  triplets.emplace_back(2, 2, 3.0);
  Eigen::SparseMatrix<double> mat{3, 3};
  mat.setFromTriplets(triplets.begin(), triplets.end());

  EXPECT_DOUBLE_EQ(21.0, sleipnir::SparseLpNorm<1>(mat));
  EXPECT_DOUBLE_EQ(std::sqrt(69.0), sleipnir::SparseLpNorm<2>(mat));
  EXPECT_DOUBLE_EQ(std::cbrt(261.0), sleipnir::SparseLpNorm<3>(mat));
  EXPECT_DOUBLE_EQ(5.0, sleipnir::SparseLpNorm<Eigen::Infinity>(mat));

  Eigen::SparseVector<double> vec{3};
  vec.insertBack(0) = -5.0;
  vec.insertBack(1) = 0.0;
  vec.insertBack(2) = 3.0;

  EXPECT_DOUBLE_EQ(8.0, sleipnir::SparseLpNorm<1>(vec));
  EXPECT_DOUBLE_EQ(std::sqrt(34.0), sleipnir::SparseLpNorm<2>(vec));
  EXPECT_DOUBLE_EQ(std::cbrt(152.0), sleipnir::SparseLpNorm<3>(vec));
  EXPECT_DOUBLE_EQ(5.0, sleipnir::SparseLpNorm<Eigen::Infinity>(vec));
}
