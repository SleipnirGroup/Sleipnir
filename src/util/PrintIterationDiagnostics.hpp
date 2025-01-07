// Copyright (c) Sleipnir contributors

#pragma once

#include <chrono>
#include <cmath>
#include <ranges>
#include <string>

#include "sleipnir/util/Print.hpp"
#include "sleipnir/util/small_vector.hpp"
#include "util/ToMilliseconds.hpp"

namespace sleipnir {

/**
 * Prints diagnostics for the current iteration.
 *
 * @param iterations Number of iterations.
 * @param feasibilityRestoration Whether solver is in feasibility restoration
 *     mode.
 * @param time The iteration duration.
 * @param error The error.
 * @param cost The cost.
 * @param infeasibility The infeasibility.
 * @param δ The Hessian regularization factor.
 * @param α The step size.
 */
template <typename Rep, typename Period = std::ratio<1>>
void PrintIterationDiagnostics(int iterations, bool feasibilityRestoration,
                               const std::chrono::duration<Rep, Period>& time,
                               double error, double cost, double infeasibility,
                               double δ, double α) {
  if (iterations % 20 == 0) {
    sleipnir::println("{:^4}   {:^9}  {:^13}  {:^13}  {:^13}  {:^4}  {:^10}",
                      "iter", "time (ms)", "error", "cost", "infeasibility",
                      "reg", "backtracks");
    sleipnir::println("{:=^79}", "");
  }

  sleipnir::print("{:4}{}  {:9.3f}  {:13e}  {:13e}  {:13e}  ", iterations,
                  feasibilityRestoration ? "r" : " ", ToMilliseconds(time),
                  error, cost, infeasibility);

  // Print regularization
  if (δ == 0.0) {
    sleipnir::print(" 0  ");
  } else {
    int exponent = std::log10(δ);

    if (exponent == 0) {
      sleipnir::print(" 1  ");
    } else if (exponent == 1) {
      sleipnir::print("10  ");
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
      for (const auto& digit : std::ranges::views::reverse(digits)) {
        reg += strs[digit];
      }

      sleipnir::print("{:<4}", reg);
    }
  }

  sleipnir::println("  {:10d}", static_cast<int>(-std::log2(α)));
}

}  // namespace sleipnir
