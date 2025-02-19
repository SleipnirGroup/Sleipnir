// Copyright (c) Sleipnir contributors

#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/variable.hpp>

#include "catch_string_converters.hpp"

TEST_CASE("Variable - Default constructor", "[Variable]") {
  sleipnir::Variable a;

  CHECK(a.value() == 0.0);
  CHECK(a.type() == sleipnir::ExpressionType::LINEAR);
}
