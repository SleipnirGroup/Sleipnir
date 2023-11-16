// Copyright (c) Sleipnir contributors

#include <algorithm>
#include <string_view>
#include <vector>

#include "CasADi.hpp"
#include "Sleipnir.hpp"
#include "Util.hpp"

int main(int argc, char* argv[]) {
  std::vector<std::string_view> args{argv + 1, argv + argc};

  constexpr bool diagnostics = false;

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
  if (args.size() == 0 ||
      std::find(args.begin(), args.end(), "--casadi") != args.end()) {
    RunBenchmarksAndLog<casadi::Opti>("flywheel-scalability-results-casadi.csv",
                                      diagnostics, T, sampleSizesToTest,
                                      &FlywheelCasADi);
  }
  if (args.size() == 0 ||
      std::find(args.begin(), args.end(), "--sleipnir") != args.end()) {
    RunBenchmarksAndLog<sleipnir::OptimizationProblem>(
        "flywheel-scalability-results-sleipnir.csv", diagnostics, T,
        sampleSizesToTest, &FlywheelSleipnir);
  }
}
