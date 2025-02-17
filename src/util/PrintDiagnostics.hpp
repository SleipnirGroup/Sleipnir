// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ranges>
#include <string>
#include <utility>

#include "sleipnir/util/Print.hpp"
#include "sleipnir/util/SetupProfiler.hpp"
#include "sleipnir/util/SolveProfiler.hpp"
#include "sleipnir/util/small_vector.hpp"

namespace sleipnir {

/**
 * Iteration type.
 */
enum class IterationType : uint8_t {
  /// Normal iteration.
  kNormal,
  /// Accepted second-order correction iteration.
  kAcceptedSOC,
  /// Rejected second-order correction iteration.
  kRejectedSOC
};

/**
 * Converts std::chrono::duration to a number of milliseconds rounded to three
 * decimals.
 */
template <typename Rep, typename Period = std::ratio<1>>
constexpr double ToMs(const std::chrono::duration<Rep, Period>& duration) {
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  return duration_cast<microseconds>(duration).count() / 1e3;
}

/**
 * Renders value as power of 10.
 *
 * @param value Value.
 */
inline std::string PowerOf10(double value) {
  if (value == 0.0) {
    return " 0";
  } else {
    int exponent = std::log10(value);

    if (exponent == 0) {
      return " 1";
    } else if (exponent == 1) {
      return "10";
    } else {
      // Gather exponent digits
      int n = std::abs(exponent);
      small_vector<int> digits;
      do {
        digits.emplace_back(n % 10);
        n /= 10;
      } while (n > 0);

      std::string output = "10";

      // Append exponent
      if (exponent < 0) {
        output += "⁻";
      }
      constexpr std::array strs = {"⁰", "¹", "²", "³", "⁴",
                                   "⁵", "⁶", "⁷", "⁸", "⁹"};
      for (const auto& digit : digits | std::views::reverse) {
        output += strs[digit];
      }

      return output;
    }
  }
}

/**
 * Prints diagnostics for the current iteration.
 *
 * @param iterations Number of iterations.
 * @param type The iteration's type.
 * @param time The iteration duration.
 * @param error The error.
 * @param cost The cost.
 * @param infeasibility The infeasibility.
 * @param complementarity The complementarity.
 * @param μ The barrier parameter.
 * @param δ The Hessian regularization factor.
 * @param primal_α The primal step size.
 * @param primal_α_max The max primal step size.
 * @param dual_α The dual step size.
 */
template <typename Rep, typename Period = std::ratio<1>>
void PrintIterationDiagnostics(int iterations, IterationType type,
                               const std::chrono::duration<Rep, Period>& time,
                               double error, double cost, double infeasibility,
                               double complementarity, double μ, double δ,
                               double primal_α, double primal_α_max,
                               double dual_α) {
  if (iterations % 20 == 0) {
    if (iterations == 0) {
      sleipnir::print("┏");
    } else {
      sleipnir::print("┢");
    }
    sleipnir::print(
        "{:━^4}┯{:━^4}┯{:━^9}┯{:━^12}┯{:━^13}┯{:━^12}┯{:━^12}┯{:━^5}┯{:━^5}┯"
        "{:━^8}┯{:━^8}┯{:━^2}",
        "", "", "", "", "", "", "", "", "", "", "", "");
    if (iterations == 0) {
      sleipnir::println("┓");
    } else {
      sleipnir::println("┪");
    }
    sleipnir::println(
        "┃{:^4}│{:^4}│{:^9}│{:^12}│{:^13}│{:^12}│{:^12}│{:^5}│{:^5}│{:^8}│{:^8}"
        "│{:^2}┃",
        "iter", "type", "time (ms)", "error", "cost", "infeas.", "complement.",
        "μ", "reg", "primal α", "dual α", "↩");
    sleipnir::println(
        "┡{:━^4}┷{:━^4}┷{:━^9}┷{:━^12}┷{:━^13}┷{:━^12}┷{:━^12}┷{:━^5}┷{:━^5}┷"
        "{:━^8}┷{:━^8}┷{:━^2}┩",
        "", "", "", "", "", "", "", "", "", "", "", "");
  }

  // For the number of backtracks, we want x such that:
  //
  //   α_max 2⁻ˣ = α
  //   2⁻ˣ = α/α_max
  //   −x = std::log2(α/α_max)
  //   x = −std::log2(α/α_max)
  int backtracks = static_cast<int>(-std::log2(primal_α / primal_α_max));

  constexpr std::array kIterationTypes = {"norm", "✓SOC", "XSOC"};
  sleipnir::println(
      "│{:4} {:4} {:9.3f} {:12e} {:13e} {:12e} {:12e} {:<5} {:<5} {:.2e} "
      "{:.2e} {:2d}│",
      iterations, kIterationTypes[std::to_underlying(type)], ToMs(time), error,
      cost, infeasibility, complementarity, PowerOf10(μ), PowerOf10(δ),
      primal_α, dual_α, backtracks);
}

/**
 * Renders histogram of the given normalized value.
 *
 * @tparam Width Width of the histogram in characters.
 * @param value Normalized value from 0 to 1.
 */
template <int Width>
  requires(Width > 0)
std::string Histogram(double value) {
  value = std::clamp(value, 0.0, 1.0);

  double ipart;
  int fpart = static_cast<int>(std::modf(value * Width, &ipart) * 8);

  constexpr std::array strs = {" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"};
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
  // Print bottom of iteration diagnostics table
  sleipnir::println("└{:─^105}┘", "");

  // Print total time
  auto setupDuration = ToMs(setupProfilers[0].Duration());
  auto solveDuration = ToMs(solveProfilers[0].TotalDuration());
  sleipnir::println("\nTime: {:.3f} ms", setupDuration + solveDuration);
  sleipnir::println("  ↳ setup: {:.3f} ms", setupDuration);
  sleipnir::println("  ↳ solve: {:.3f} ms ({} iterations)", solveDuration,
                    iterations);

  // Print setup diagnostics
  sleipnir::println("\n┏{:━^21}┯{:━^18}┯{:━^10}┓", "", "", "");
  sleipnir::println("┃{:^21}│{:^18}│{:^10}┃", "trace", "percent", "total (ms)");
  sleipnir::println("┡{:━^21}┷{:━^18}┷{:━^10}┩", "", "", "");

  for (auto& profiler : setupProfilers) {
    double norm = setupDuration == 0.0
                      ? (&profiler == &setupProfilers[0] ? 1.0 : 0.0)
                      : ToMs(profiler.Duration()) / setupDuration;
    sleipnir::println("│{:<21} {:>6.2f}%▕{}▏ {:>10.3f}│", profiler.name,
                      norm * 100.0, Histogram<9>(norm),
                      ToMs(profiler.Duration()));
  }

  sleipnir::println("└{:─^51}┘", "");

  // Print solve diagnostics
  sleipnir::println("┏{:━^21}┯{:━^18}┯{:━^10}┯{:━^9}┯{:━^4}┓", "", "", "", "",
                    "");
  sleipnir::println("┃{:^21}│{:^18}│{:^10}│{:^9}│{:^4}┃", "trace", "percent",
                    "total (ms)", "each (ms)", "runs");
  sleipnir::println("┡{:━^21}┷{:━^18}┷{:━^10}┷{:━^9}┷{:━^4}┩", "", "", "", "",
                    "");

  for (auto& profiler : solveProfilers) {
    double norm = solveDuration == 0.0
                      ? (&profiler == &solveProfilers[0] ? 1.0 : 0.0)
                      : ToMs(profiler.TotalDuration()) / solveDuration;
    sleipnir::println("│{:<21} {:>6.2f}%▕{}▏ {:>10.3f} {:>9.3f} {:>4}│",
                      profiler.name, norm * 100.0, Histogram<9>(norm),
                      ToMs(profiler.TotalDuration()),
                      ToMs(profiler.AverageDuration()), profiler.NumSolves());
  }

  sleipnir::println("└{:─^66}┘\n", "");
}

}  // namespace sleipnir
