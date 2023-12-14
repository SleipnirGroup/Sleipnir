// Copyright (c) Sleipnir contributors

#include "CmdlineArguments.hpp"

#include <algorithm>

namespace {
static std::span<char*> args;
}  // namespace

void SetCmdlineArgs(char* argv[], int argc) {
  args = std::span(argv, argc);
}

std::span<char*> GetCmdlineArgs() {
  return args;
}

bool CmdlineArgPresent(std::string_view arg) {
  return std::find(args.begin(), args.end(), arg) != args.end();
}
