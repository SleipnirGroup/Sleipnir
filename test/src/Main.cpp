// Copyright (c) Sleipnir contributors

#include <catch2/catch_session.hpp>

#include "CmdlineArguments.hpp"

int main(int argc, char* argv[]) {
  Argv() = CmdlineArgs(argv, argc);

  Catch::Session session;

  [[maybe_unused]] bool enableDiagnostics;
  auto cli = session.cli() |
             Catch::Clara::Opt(enableDiagnostics)["--enable-diagnostics"](
                 "enables solver diagnostic prints");
  session.cli(cli);
  int ret = session.applyCommandLine(argc, argv);
  if (ret != 0) {
    return ret;
  }
  session.configData().allowZeroTests = true;

  return session.run();
}
