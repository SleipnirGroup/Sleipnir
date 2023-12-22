// Copyright (c) Sleipnir contributors

#include "CmdlineArguments.hpp"

namespace {
static CmdlineArgs args;
}  // namespace

CmdlineArgs& Argv() {
  return args;
}
