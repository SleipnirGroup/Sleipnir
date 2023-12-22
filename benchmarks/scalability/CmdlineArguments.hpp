// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <span>
#include <string_view>

class CmdlineArgs {
 public:
  CmdlineArgs() = default;

  /**
   * Populate commandline arguments.
   *
   * @param argv argv argument of main(int argc, char* argv[]).
   * @param argc argc argument of main(int argc, char* argv[]).
   */
  CmdlineArgs(char* argv[], int argc) : args(argv, argc) {}

  /**
   * Returns size of the list of arguments.
   */
  size_t Size() const { return args.size(); }

  /**
   * Returns true if the given argument is present in the test executable's
   * commandline arguments.
   */
  bool Contains(std::string_view arg) const {
    return std::find(args.begin(), args.end(), arg) != args.end();
  }

 private:
  std::span<char*> args;
};
