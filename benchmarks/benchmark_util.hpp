// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <chrono>
#include <format>
#include <span>
#include <string_view>
#include <vector>

#include <benchmark/benchmark.h>

/// Adds minimum and maximum to the statistics computed over a benchmark's
/// repetitions.
///
/// @param b The benchmark.
inline void add_min_max_statistics(benchmark::Benchmark* b) {
  b->ComputeStatistics(
      "min", [](const std::vector<double>& v) { return std::ranges::min(v); });
  b->ComputeStatistics(
      "max", [](const std::vector<double>& v) { return std::ranges::max(v); });
}

/// Registers setup and solve benchmarks for an optimization problem.
///
/// The benchmarks are named "<library>/Setup/<N>" and "<library>/Solve/<N>",
/// where N is the number of samples. The scale of the problem is increased by
/// increasing the number of timesteps within the time horizon.
///
/// @param library The optimization library's name.
/// @param nums_samples The numbers of samples for which to record results.
/// @param setup A function that takes a timestep duration and number of samples
///     and returns an optimization problem instance.
/// @param solve A function that takes an optimization problem instance by
///     reference, solves it, and returns true if the solve succeeded.
template <typename Setup, typename Solve>
void register_optimization_benchmarks(std::string_view library,
                                      std::span<const int> nums_samples,
                                      Setup setup, Solve solve) {
  using namespace std::chrono_literals;

  // Time horizon of the optimization problem
  constexpr std::chrono::duration<double> T = 5s;

  // Sample counts for which to record results
  auto args = [=](benchmark::Benchmark* b) {
    for (int N : nums_samples) {
      b->Arg(N);
    }
  };

  // Setup time includes building the problem
  benchmark::RegisterBenchmark(
      std::format("{}/Setup", library),
      [=](benchmark::State& state) {
        int N = static_cast<int>(state.range(0));
        for (auto _ : state) {
          auto start_time = std::chrono::steady_clock::now();
          auto problem = setup(T / N, N);
          benchmark::DoNotOptimize(problem);
          auto end_time = std::chrono::steady_clock::now();
          state.SetIterationTime(
              std::chrono::duration<double>{end_time - start_time}.count());
        }
      })
      ->Apply(args)
      ->Apply(add_min_max_statistics)
      ->UseManualTime()
      ->Unit(benchmark::kMillisecond);

  // Solve time is the time to solve a freshly built problem
  benchmark::RegisterBenchmark(
      std::format("{}/Solve", library),
      [=](benchmark::State& state) {
        int N = static_cast<int>(state.range(0));
        for (auto _ : state) {
          auto problem = setup(T / N, N);
          auto start_time = std::chrono::steady_clock::now();
          bool success = solve(problem);
          auto end_time = std::chrono::steady_clock::now();
          if (!success) {
            state.SkipWithError("Solve failed");
            break;
          }
          state.SetIterationTime(
              std::chrono::duration<double>{end_time - start_time}.count());
        }
      })
      ->Apply(args)
      ->Apply(add_min_max_statistics)
      ->UseManualTime()
      ->Unit(benchmark::kMillisecond);
}
