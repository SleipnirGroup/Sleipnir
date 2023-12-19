// Copyright (c) Sleipnir contributors

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
  for (int N = 100; N < 300; N += 100) {
    sampleSizesToTest.emplace_back(N);
  }
  sampleSizesToTest.emplace_back(300);

  fmt::print("Solving cart-pole problem from N = {} to N = {}.\n",
             sampleSizesToTest.front(), sampleSizesToTest.back());
  if (args.size() == 0 ||
      std::find(args.begin(), args.end(), "--casadi") != args.end()) {
    RunBenchmarksAndLog<casadi::Opti>(
        "cart-pole-scalability-results-casadi.csv", diagnostics, T,
        sampleSizesToTest, &CartPoleCasADi);
  }
  if (args.size() == 0 ||
      std::find(args.begin(), args.end(), "--sleipnir") != args.end()) {
    RunBenchmarksAndLog<sleipnir::OptimizationProblem>(
        "cart-pole-scalability-results-sleipnir.csv", diagnostics, T,
        sampleSizesToTest, &CartPoleSleipnir);
  }
}
