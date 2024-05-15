// Copyright (c) Sleipnir contributors

#include <array>
#include <print>

#include "CurrentManager.hpp"

#ifndef RUNNING_TESTS
int main() {
  CurrentManager manager{std::array{1.0, 5.0, 10.0, 5.0}, 40.0};

  auto currents = manager.Calculate(std::array{25.0, 10.0, 5.0, 0.0});

  std::print("currents = [");
  for (size_t i = 0; i < currents.size(); ++i) {
    std::print("{}", currents[i]);
    if (i < currents.size() - 1) {
      std::print(", ");
    }
  }
  std::println("]");
}
#endif
