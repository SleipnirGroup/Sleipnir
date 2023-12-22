// Copyright (c) Sleipnir contributors

#include <vector>

#include "CasADi.hpp"
#include "CmdlineArguments.hpp"
#include "Sleipnir.hpp"
#include "Util.hpp"

int main(int argc, char* argv[]) {
  CmdlineArgs args{argv, argc};

  bool runCasadi = args.Contains("--casadi");
  bool runSleipnir = args.Contains("--sleipnir");
  if (!runCasadi && !runSleipnir) {
    runCasadi = true;
    runSleipnir = true;
  }
  bool diagnostics = args.Contains("--enable-diagnostics");

  constexpr auto T = 5_s;

  std::vector<int> sampleSizesToTest;
  for (int N = 100; N < 1000; N += 100) {
    sampleSizesToTest.emplace_back(N);
  }
  for (int N = 1000; N < 5000; N += 1000) {
    sampleSizesToTest.emplace_back(N);
  }
  sampleSizesToTest.emplace_back(5000);

  fmt::print("Solving flywheel problem from N = {} to N = {}.\n",
             sampleSizesToTest.front(), sampleSizesToTest.back());
  if (runCasadi) {
    RunBenchmarksAndLog<casadi::Opti>("flywheel-scalability-results-casadi.csv",
                                      diagnostics, T, sampleSizesToTest,
                                      &FlywheelCasADi);
  }
  if (runSleipnir) {
    RunBenchmarksAndLog<sleipnir::OptimizationProblem>(
        "flywheel-scalability-results-sleipnir.csv", diagnostics, T,
        sampleSizesToTest, &FlywheelSleipnir);
  }
}
