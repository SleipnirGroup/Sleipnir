// Copyright (c) Sleipnir contributors

#include <gtest/gtest.h>

#include "CmdlineArguments.hpp"

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);

  SetCmdlineArgs(argv, argc);

  return RUN_ALL_TESTS();
}
