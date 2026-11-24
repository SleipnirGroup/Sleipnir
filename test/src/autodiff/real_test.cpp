// Copyright (c) Sleipnir contributors

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/autodiff/real.hpp>

#include "catch_string_converters.hpp"
#include "scalar_types.hpp"

TEMPLATE_TEST_CASE("Real - Default constructor", "[Real]", SCALAR_TYPES) {
  using T = TestType;

  slp::Real<T> a;
  CHECK(a.value() == T(0));
  CHECK(a.type() == slp::ExpressionType::LINEAR);
}

TEMPLATE_TEST_CASE("Real - Constant constructor", "[Real]", SCALAR_TYPES) {
  using T = TestType;

  // float
  slp::Real<T> a{1.0};
  CHECK(a.value() == T(1));
  CHECK(a.type() == slp::ExpressionType::CONSTANT);

  // int
  slp::Real<T> b{2};
  CHECK(b.value() == T(2));
  CHECK(b.type() == slp::ExpressionType::CONSTANT);
}

TEMPLATE_TEST_CASE("Real - Set value", "[Real]", SCALAR_TYPES) {
  using T = TestType;

  slp::Real<T> a;

  a.set_value(T(1));
  CHECK(a.value() == T(1));

  a.set_value(T(2));
  CHECK(a.value() == T(2));
}
