// Copyright (c) Sleipnir contributors

#include <catch2/catch_test_macros.hpp>

#include "util/ScopeExit.hpp"

TEST_CASE("scope_exit - Scope exit", "[scope_exit]") {
  int exitCount = 0;

  {
    sleipnir::scope_exit exit{[&] { ++exitCount; }};

    CHECK(exitCount == 0);
  }

  CHECK(exitCount == 1);
}

TEST_CASE("scope_exit - Release", "[scope_exit]") {
  int exitCount = 0;

  {
    sleipnir::scope_exit exit1{[&] { ++exitCount; }};
    sleipnir::scope_exit exit2 = std::move(exit1);
    sleipnir::scope_exit exit3 =
        std::move(exit1);  // NOLINT (clang-analyzer-cplusplus.Move)
    CHECK(exitCount == 0);
  }
  CHECK(exitCount == 1);

  {
    sleipnir::scope_exit exit{[&] { ++exitCount; }};
    exit.release();
  }
  CHECK(exitCount == 1);
}
