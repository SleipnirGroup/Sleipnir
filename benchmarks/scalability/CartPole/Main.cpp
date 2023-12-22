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
  for (int N = 100; N < 300; N += 100) {
    sampleSizesToTest.emplace_back(N);
  }
  sampleSizesToTest.emplace_back(300);

  fmt::print("Solving cart-pole problem from N = {} to N = {}.\n",
             sampleSizesToTest.front(), sampleSizesToTest.back());
  if (runCasadi) {
    RunBenchmarksAndLog<casadi::Opti>(
        "cart-pole-scalability-results-casadi.csv", diagnostics, T,
        sampleSizesToTest, &CartPoleCasADi);
  }
  if (runSleipnir) {
    RunBenchmarksAndLog<sleipnir::OptimizationProblem>(
        "cart-pole-scalability-results-sleipnir.csv", diagnostics, T,
        sampleSizesToTest, &CartPoleSleipnir);
  }
}
