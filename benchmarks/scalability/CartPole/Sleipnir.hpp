// Copyright (c) Sleipnir contributors

#pragma once

#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <units/time.h>

/**
 * Creates a cart-pole nonlinear optimization problem with Sleipnir.
 *
 * @param dt Timestep duration.
 * @param N Number of samples in the problem.
 */
sleipnir::OptimizationProblem CartPoleSleipnir(units::second_t dt, int N);
