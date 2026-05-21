// Copyright (c) Sleipnir contributors

#include <format>

#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/solver/util/inertia.hpp>

TEST_CASE("Inertia - Formatter", "[Formatter]") {
  CHECK(std::format("{}", slp::Inertia{1, -2, 0}) == "(1, -2, 0)");
  CHECK(std::format("{:+}", slp::Inertia{1, -2, 0}) == "(+1, -2, +0)");
  CHECK(std::format("{: }", slp::Inertia{1, -2, 0}) == "( 1, -2,  0)");
  CHECK(std::format("{: >2}", slp::Inertia{1, 20, 0}) == "( 1, 20,  0)");
}
