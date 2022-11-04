// Copyright (c) Joshua Nichols and Tyler Veness

#include <array>

#include <gtest/gtest.h>

#include "CurrentManager.h"

TEST(CurrentManagerTest, EnoughCurrent) {
  CurrentManager manager{std::array{1_A, 5_A, 10_A, 5_A}, 40_A};

  auto currents = manager.Calculate(std::array{25_A, 10_A, 5_A, 0_A});

  EXPECT_NEAR(25.0, currents[0].value(), 1e-2);
  EXPECT_NEAR(10.0, currents[1].value(), 1e-2);
  EXPECT_NEAR(5.0, currents[2].value(), 1e-2);
  EXPECT_NEAR(0.0, currents[3].value(), 1e-2);
}

TEST(CurrentManagerTest, NotEnoughCurrent) {
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
  EXPECT_NEAR(29.960, currents[0].value(), 1e-2);
  EXPECT_NEAR(9.00793, currents[1].value(), 1e-2);
  EXPECT_NEAR(1.0317, currents[2].value(), 1e-2);
  EXPECT_NEAR(1.454e-09, currents[3].value(), 1e-2);
}
