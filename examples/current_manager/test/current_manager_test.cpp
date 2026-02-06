// Copyright (c) Sleipnir contributors

#include "current_manager.hpp"

#include <array>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;

TEST_CASE("CurrentManager - Enough current", "[CurrentManager]") {
  CurrentManager manager{std::array{1.0, 5.0, 10.0, 5.0}, 40.0};

  auto currents = manager.calculate(std::array{25.0, 10.0, 5.0, 0.0});

  CHECK_THAT(currents[0], WithinAbs(25.0, 1e-3));
  CHECK_THAT(currents[1], WithinAbs(10.0, 1e-3));
  CHECK_THAT(currents[2], WithinAbs(5.0, 1e-3));
  CHECK_THAT(currents[3], WithinAbs(0.0, 1e-3));
}

TEST_CASE("CurrentManager - Not enough current", "[CurrentManager]") {
  CurrentManager manager{std::array{1.0, 5.0, 10.0, 5.0}, 40.0};

  auto currents = manager.calculate(std::array{30.0, 10.0, 5.0, 0.0});

  // Expected values are from the following program:
  //
  // #!/usr/bin/env python3
  //
  // from scipy.optimize import minimize
  //
  // r = [30.0, 10.0, 5.0, 0.0]
  // q = [1.0, 5.0, 10.0, 5.0]
  //
  // result = minimize(
  //     lambda x: sum((r[i] - x[i]) ** 2 / q[i] ** 2 for i in range(4)),
  //     [0.0, 0.0, 0.0, 0.0],
  //     constraints=[
  //         {"type": "ineq", "fun": lambda x: x},
  //         {"type": "ineq", "fun": lambda x: 40.0 - sum(x)},
  //     ],
  // )
  // print(result.x)
  CHECK_THAT(currents[0], WithinAbs(29.960, 1e-3));
  CHECK_THAT(currents[1], WithinAbs(9.008, 1e-3));
  CHECK_THAT(currents[2], WithinAbs(1.032, 1e-3));
  CHECK_THAT(currents[3], WithinAbs(0.0, 1e-3));
}
