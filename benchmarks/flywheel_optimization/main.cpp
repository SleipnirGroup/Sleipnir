// Copyright (c) Sleipnir contributors

#include <algorithm>
#include <array>
#include <string_view>

#include <benchmark/benchmark.h>

#include "casadi.hpp"
#include "sleipnir.hpp"

int main(int argc, char* argv[]) {
  benchmark::Initialize(&argc, argv);

  // Remove flags Google Benchmark doesn't recognize after processing them
  int new_argc = std::remove(argv + 1, argv + argc,
                             std::string_view{"--enable-diagnostics"}) -
                 argv;
  bool diagnostics = new_argc < argc;
  argc = new_argc;

  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }

  // Sample counts for which to record results
  constexpr std::array nums_samples{100, 200, 300,  400,  500,  600,  700,
                                    800, 900, 1000, 2000, 3000, 4000, 5000};

  register_casadi_benchmarks(nums_samples, diagnostics);
  register_sleipnir_benchmarks(nums_samples, diagnostics);

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}
