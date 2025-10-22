// Copyright (c) Sleipnir contributors

#pragma once

#include <concepts>
#include <ranges>

template <std::floating_point T>
auto float_range(T start, T end, T step) {
  return std::views::iota(0, static_cast<int>((end - start) / step)) |
         std::views::transform([=](auto&& i) { return start + i * step; });
}
