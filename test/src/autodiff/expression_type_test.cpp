// Copyright (c) Sleipnir contributors

#include <format>

#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/expression_type.hpp>

TEST_CASE("ExpressionType - Formatter", "[Formatter]") {
  CHECK(std::format("{}", slp::ExpressionType::NONE) == "none");
  CHECK(std::format("{}", slp::ExpressionType::CONSTANT) == "constant");
  CHECK(std::format("{}", slp::ExpressionType::LINEAR) == "linear");
  CHECK(std::format("{}", slp::ExpressionType::QUADRATIC) == "quadratic");
  CHECK(std::format("{}", slp::ExpressionType::NONLINEAR) == "nonlinear");
}
