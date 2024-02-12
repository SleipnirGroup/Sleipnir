// Copyright (c) Sleipnir contributors

#include <array>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "CurrentManager.hpp"

TEST_CASE("CurrentManager - Enough current", "[CurrentManager]") {
  CurrentManager manager{std::array{1_A, 5_A, 10_A, 5_A}, 40_A};

  auto currents = manager.Calculate(std::array{25_A, 10_A, 5_A, 0_A});

  CHECK(currents[0].value() == Catch::Approx(25.0).margin(1e-3));
  CHECK(currents[1].value() == Catch::Approx(10.0).margin(1e-3));
  CHECK(currents[2].value() == Catch::Approx(5.0).margin(1e-3));
  CHECK(currents[3].value() == Catch::Approx(0.0).margin(1e-3));
}

TEST_CASE("CurrentManager - Not enough current", "[CurrentManager]") {
  CurrentManager manager{std::array{1_A, 5_A, 10_A, 5_A}, 40_A};

  auto currents = manager.Calculate(std::array{30_A, 10_A, 5_A, 0_A});

  // Expected values are from the following CasADi program:
  //
  // #!/usr/bin/env python3
  //
  // import casadi as ca
  // import numpy as np
  //
  // opti = ca.Opti()
  // allocated_currents = opti.variable(4, 1)
  //
  // current_tolerances = np.array([[1.0], [5.0], [10.0], [5.0]])
  // desired_currents = np.array([[30.0], [10.0], [5.0], [0.0]])
  //
  // J = 0.0
  // current_sum = 0.0
  // for i in range(4):
  //     error = desired_currents[i, 0] - allocated_currents[i, 0]
  //     J += error**2 / current_tolerances[i] ** 2
  //
  //     current_sum += allocated_currents[i, 0]
  //
  //     # Currents must be nonnegative
  //     opti.subject_to(allocated_currents[i, 0] >= 0.0)
  // opti.minimize(J)
  //
  // # Keep total current below maximum
  // opti.subject_to(current_sum <= 40.0)
  //
  // opti.solver("ipopt")
  // print(opti.solve().value(allocated_currents))
  CHECK(currents[0].value() == Catch::Approx(29.960).margin(1e-3));
  CHECK(currents[1].value() == Catch::Approx(9.007).margin(1e-3));
  CHECK(currents[2].value() == Catch::Approx(1.032).margin(1e-3));
  CHECK(currents[3].value() == Catch::Approx(0.0).margin(1e-3));
}
