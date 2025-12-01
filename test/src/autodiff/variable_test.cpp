// Copyright (c) Sleipnir contributors

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/variable.hpp>

#include "catch_string_converters.hpp"
#include "scalar_types_under_test.hpp"

TEMPLATE_TEST_CASE("Variable - Default constructor", "[Variable]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Variable<T> a;
  CHECK(a.value() == T(0));
  CHECK(a.type() == slp::ExpressionType::LINEAR);
}

TEMPLATE_TEST_CASE("Variable - Constant constructor", "[Variable]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  // float
  slp::Variable<T> a{1.0};
  CHECK(a.value() == T(1));
  CHECK(a.type() == slp::ExpressionType::CONSTANT);

  // int
  slp::Variable<T> b{2};
  CHECK(b.value() == T(2));
  CHECK(b.type() == slp::ExpressionType::CONSTANT);
}

TEMPLATE_TEST_CASE("Variable - Set value", "[Variable]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Variable<T> a;

  a.set_value(T(1));
  CHECK(a.value() == T(1));

  a.set_value(T(2));
  CHECK(a.value() == T(2));
}
