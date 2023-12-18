// Copyright (c) Sleipnir contributors

#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <sleipnir/util/Formatters.hpp>

TEST(FormattersTest, Eigen) {
  Eigen::Matrix<double, 3, 2> A{{0.0, 1.0}, {2.0, 3.0}, {4.0, 5.0}};
  EXPECT_EQ(
      "  0.000000  1.000000\n"
      "  2.000000  3.000000\n"
      "  4.000000  5.000000",
      fmt::format("{:f}", A));

  Eigen::MatrixXd B{{0.0, 1.0}, {2.0, 3.0}, {4.0, 5.0}};
  EXPECT_EQ(
      "  0.000000  1.000000\n"
      "  2.000000  3.000000\n"
      "  4.000000  5.000000",
      fmt::format("{:f}", B));

  Eigen::SparseMatrix<double> C{3, 2};
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.emplace_back(0, 1, 1.0);
  triplets.emplace_back(1, 0, 2.0);
  triplets.emplace_back(1, 1, 3.0);
  triplets.emplace_back(2, 0, 4.0);
  triplets.emplace_back(2, 1, 5.0);
  C.setFromTriplets(triplets.begin(), triplets.end());
  EXPECT_EQ(
      "  0.000000  1.000000\n"
      "  2.000000  3.000000\n"
      "  4.000000  5.000000",
      fmt::format("{:f}", C));
}

TEST(FormattersTest, Variable) {
  EXPECT_EQ("4.000000", fmt::format("{:f}", sleipnir::Variable{4.0}));
}

TEST(FormattersTest, VariableMatrix) {
  Eigen::Matrix<double, 3, 2> A{{0.0, 1.0}, {2.0, 3.0}, {4.0, 5.0}};

  sleipnir::VariableMatrix B{3, 2};
  B = A;
  EXPECT_EQ(
      "  0.000000  1.000000\n"
      "  2.000000  3.000000\n"
      "  4.000000  5.000000",
      fmt::format("{:f}", B));

  sleipnir::VariableBlock<sleipnir::VariableMatrix> C{B};
  EXPECT_EQ(
      "  0.000000  1.000000\n"
      "  2.000000  3.000000\n"
      "  4.000000  5.000000",
      fmt::format("{:f}", C));
}
