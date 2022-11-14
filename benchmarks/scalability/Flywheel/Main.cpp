// Copyright (c) Joshua Nichols and Tyler Veness

#include <vector>

#include "CasADi.hpp"
#include "Sleipnir.hpp"
#include "Util.hpp"

int main() {
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
  return RunBenchmarksAndLog("flywheel-scalability-results.csv", diagnostics, T,
                             sampleSizesToTest, &FlywheelCasADi,
                             &FlywheelSleipnir);
}
