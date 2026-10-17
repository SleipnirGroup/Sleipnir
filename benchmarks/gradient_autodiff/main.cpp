// Copyright (c) Sleipnir contributors

#include <array>

#include <benchmark/benchmark.h>

#include "casadi.hpp"
#include "sleipnir.hpp"

int main(int argc, char* argv[]) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }

  // Variable counts for which to record results
  constexpr std::array nums_variables{1000, 2000, 3000, 4000, 5000,
                                      6000, 7000, 8000, 9000, 10000};

  register_casadi_benchmarks(nums_variables);
  register_sleipnir_benchmarks(nums_variables);

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}
