// Copyright (c) Sleipnir contributors

#pragma once

#include <chrono>

namespace sleipnir {

/**
 * Converts std::chrono::duration to a number of milliseconds rounded to three
 * decimals.
 */
template <typename Rep, typename Period = std::ratio<1>>
double ToMilliseconds(const std::chrono::duration<Rep, Period>& duration) {
  using std::chrono::duration_cast;
  using std::chrono::nanoseconds;
  return duration_cast<nanoseconds>(duration).count() / 1e6;
}

}  // namespace sleipnir
