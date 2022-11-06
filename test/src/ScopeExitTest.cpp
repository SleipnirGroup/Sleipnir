// Copyright (c) Joshua Nichols and Tyler Veness

#include <gtest/gtest.h>

#include "ScopeExit.h"

TEST(ScopeExitTest, ScopeExit) {
  int exitCount = 0;

  {
    sleipnir::scope_exit exit{[&] { ++exitCount; }};

    EXPECT_EQ(0, exitCount);
  }

  EXPECT_EQ(1, exitCount);
}

TEST(ScopeExitTest, Release) {
  int exitCount = 0;

  {
    sleipnir::scope_exit exit1{[&] { ++exitCount; }};
    sleipnir::scope_exit exit2 = std::move(exit1);
    sleipnir::scope_exit exit3 = std::move(exit1);
    EXPECT_EQ(0, exitCount);
  }
  EXPECT_EQ(1, exitCount);

  {
    sleipnir::scope_exit exit{[&] { ++exitCount; }};
    exit.release();
  }
  EXPECT_EQ(1, exitCount);
}
