// Copyright (c) Joshua Nichols and Tyler Veness

#include <array>

#include <fmt/core.h>
#include <units/formatter.h>

#include "CurrentManager.hpp"

#ifndef RUNNING_TESTS
int main() {
  CurrentManager manager{std::array{1_A, 5_A, 10_A, 5_A}, 40_A};

  auto currents = manager.Calculate(std::array{25_A, 10_A, 5_A, 0_A});

  fmt::print("currents = [");
  for (size_t i = 0; i < currents.size(); ++i) {
    fmt::print("{}", currents[i]);
    if (i < currents.size() - 1) {
      fmt::print(", ");
    }
  }
  fmt::print("]\n");
}
#endif
