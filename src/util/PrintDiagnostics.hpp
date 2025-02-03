// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ranges>
#include <string>
#include <utility>

#include "sleipnir/util/Print.hpp"
#include "sleipnir/util/SetupProfiler.hpp"
#include "sleipnir/util/SolveProfiler.hpp"
#include "sleipnir/util/small_vector.hpp"
#include "util/ToMs.hpp"

namespace sleipnir {

/**
 * Iteration mode.
 */
enum class IterationMode : uint8_t {
  /// Normal iteration.
  kNormal,
  /// Second-order correction iteration.
  kSecondOrderCorrection
};

/**
 * Prints diagnostics for the current iteration.
 *
 * @param iterations Number of iterations.
 * @param mode Which mode the iteration was in.
 * @param time The iteration duration.
 * @param error The error.
 * @param cost The cost.
 * @param infeasibility The infeasibility.
 * @param δ The Hessian regularization factor.
 * @param primal_α The primal step size.
 * @param dual_α The dual step size.
 */
template <typename Rep, typename Period = std::ratio<1>>
void PrintIterationDiagnostics(int iterations, IterationMode mode,
                               const std::chrono::duration<Rep, Period>& time,
                               double error, double cost, double infeasibility,
                               double δ, double primal_α, double dual_α) {
  if (iterations % 20 == 0) {
    sleipnir::println(
        "{:^4}   {:^9}  {:^13}  {:^13}  {:^13}  {:^5}  {:^8}  {:^8}  {:^6}",
        "iter", "time ms", "error", "cost", "infeasibility", "reg", "primal α",
        "dual α", "bktrks");
    sleipnir::println("{:=^96}", "");
  }

  constexpr const char* kIterationModes[] = {" ", "s"};
  sleipnir::print("{:4}{}  {:9.3f}  {:13e}  {:13e}  {:13e}  ", iterations,
                  kIterationModes[std::to_underlying(mode)], ToMs(time), error,
                  cost, infeasibility);

  // Print regularization
  if (δ == 0.0) {
    sleipnir::print(" 0   ");
  } else {
    int exponent = std::log10(δ);

    if (exponent == 0) {
      sleipnir::print(" 1   ");
    } else if (exponent == 1) {
      sleipnir::print("10   ");
    } else {
      // Gather regularization exponent digits
      int n = std::abs(exponent);
      small_vector<int> digits;
      do {
        digits.emplace_back(n % 10);
        n /= 10;
      } while (n > 0);

      std::string reg = "10";

      // Append regularization exponent
      if (exponent < 0) {
        reg += "⁻";
      }
      constexpr const char* strs[] = {"⁰", "¹", "²", "³", "⁴",
                                      "⁵", "⁶", "⁷", "⁸", "⁹"};
      for (const auto& digit : digits | std::views::reverse) {
        reg += strs[digit];
      }

      sleipnir::print("{:<5}", reg);
    }
  }

  // Print step sizes and number of backtracks
  sleipnir::println("  {:.2e}  {:.2e}  {:6d}", primal_α, dual_α,
                    static_cast<int>(-std::log2(primal_α)));
}

/**
 * Renders histogram of the given normalized value.
 *
 * @tparam Width Width of the histogram in characters.
 * @param value Normalized value from 0 to 1.
 */
template <int Width>
  requires(Width > 0)
inline std::string Histogram(double value) {
  value = std::clamp(value, 0.0, 1.0);

  double ipart;
  int fpart = static_cast<int>(std::modf(value * Width, &ipart) * 8);

  constexpr const char* strs[] = {" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"};
  std::string hist;

  int index = 0;
  while (index < ipart) {
    hist += strs[8];
    ++index;
  }
  if (fpart > 0) {
    hist += strs[fpart];
    ++index;
  }
  while (index < Width) {
    hist += strs[0];
    ++index;
  }

  return hist;
}

/**
 * Prints final diagnostics.
 *
 * @param iterations Number of iterations.
 * @param setupProfilers Setup profilers.
 * @param solveProfilers Solve profilers.
 */
inline void PrintFinalDiagnostics(
    int iterations, const small_vector<SetupProfiler>& setupProfilers,
    const small_vector<SolveProfiler>& solveProfilers) {
  // Print total time
  auto setupDuration = ToMs(setupProfilers[0].Duration());
  auto solveDuration = ToMs(solveProfilers[0].TotalDuration());
  sleipnir::println("\nTime: {:.3f} ms", setupDuration + solveDuration);
  sleipnir::println("  ↳ setup: {:.3f} ms", setupDuration);
  sleipnir::println("  ↳ solve: {:.3f} ms ({} iterations)", solveDuration,
                    iterations);

  // Print setup diagnostics
  sleipnir::println("\n{:^21}  {:^18}{:>9}", "trace", "", "total ms");
  sleipnir::println("{:=^50}", "");

  for (auto& profiler : setupProfilers) {
    double norm =
        ToMs(profiler.Duration()) / ToMs(setupProfilers[0].Duration());
    sleipnir::println("{:<21}  {:>6.2f}%▕{}▏{:>9.3f}", profiler.name,
                      norm * 100.0, Histogram<9>(norm),
                      ToMs(profiler.Duration()));
  }

  // Print solve diagnostics
  sleipnir::println("\n{:^21}  {:^18}{:>9}  {:>9}  {:>4}", "trace", "",
                    "total ms", "each ms", "num");
  sleipnir::println("{:=^67}", "");

  for (auto& profiler : solveProfilers) {
    double norm = ToMs(profiler.TotalDuration()) /
                  ToMs(solveProfilers[0].TotalDuration());
    sleipnir::println("{:<21}  {:>6.2f}%▕{}▏{:>9.3f}  {:>9.3f}  {:>4}",
                      profiler.name, norm * 100.0, Histogram<9>(norm),
                      ToMs(profiler.TotalDuration()),
                      ToMs(profiler.AverageDuration()), profiler.NumSolves());
  }

  sleipnir::println("");
}

}  // namespace sleipnir
