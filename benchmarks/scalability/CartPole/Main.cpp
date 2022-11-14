// Copyright (c) Joshua Nichols and Tyler Veness

#include <vector>

#include "CasADi.hpp"
#include "Sleipnir.hpp"
#include "Util.hpp"

int main() {
  constexpr bool diagnostics = false;

  constexpr auto T = 5_s;

  std::vector<int> sampleSizesToTest;
  for (int N = 100; N < 500; N += 100) {
    sampleSizesToTest.emplace_back(N);
  }
  sampleSizesToTest.emplace_back(500);

  fmt::print("Solving cart-pole problem from N = {} to N = {}.\n",
             sampleSizesToTest.front(), sampleSizesToTest.back());
  return RunBenchmarksAndLog("cart-pole-scalability-results.csv", diagnostics,
                             T, sampleSizesToTest, &CartPoleCasADi,
                             &CartPoleSleipnir);
}
