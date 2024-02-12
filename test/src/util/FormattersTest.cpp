// Copyright (c) Sleipnir contributors

#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <sleipnir/util/Formatters.hpp>

TEST_CASE("Formatters - Eigen", "[formatters]") {
  Eigen::Matrix<double, 3, 2> A{{0.0, 1.0}, {2.0, 3.0}, {4.0, 5.0}};
  CHECK(fmt::format("{:f}", A) ==
        "  0.000000  1.000000\n"
        "  2.000000  3.000000\n"
        "  4.000000  5.000000");

  Eigen::MatrixXd B{{0.0, 1.0}, {2.0, 3.0}, {4.0, 5.0}};
  CHECK(fmt::format("{:f}", B) ==
        "  0.000000  1.000000\n"
        "  2.000000  3.000000\n"
        "  4.000000  5.000000");

  Eigen::SparseMatrix<double> C{3, 2};
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.emplace_back(0, 1, 1.0);
  triplets.emplace_back(1, 0, 2.0);
  triplets.emplace_back(1, 1, 3.0);
  triplets.emplace_back(2, 0, 4.0);
  triplets.emplace_back(2, 1, 5.0);
  C.setFromTriplets(triplets.begin(), triplets.end());
  CHECK(fmt::format("{:f}", C) ==
        "  0.000000  1.000000\n"
        "  2.000000  3.000000\n"
        "  4.000000  5.000000");
}

TEST_CASE("Formatters - Variable", "[formatters]") {
  CHECK(fmt::format("{:f}", sleipnir::Variable{4.0}) == "4.000000");
}

TEST_CASE("Formatters - VariableMatrix", "[formatters]") {
  Eigen::Matrix<double, 3, 2> A{{0.0, 1.0}, {2.0, 3.0}, {4.0, 5.0}};

  sleipnir::VariableMatrix B{3, 2};
  B = A;
  CHECK(fmt::format("{:f}", B) ==
        "  0.000000  1.000000\n"
        "  2.000000  3.000000\n"
        "  4.000000  5.000000");

  sleipnir::VariableBlock<sleipnir::VariableMatrix> C{B};
  CHECK(fmt::format("{:f}", C) ==
        "  0.000000  1.000000\n"
        "  2.000000  3.000000\n"
        "  4.000000  5.000000");
}
