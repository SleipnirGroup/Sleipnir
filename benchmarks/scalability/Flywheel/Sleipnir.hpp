// Copyright (c) Sleipnir contributors

#pragma once

#include <chrono>

#include <sleipnir/optimization/OptimizationProblem.hpp>

/**
 * Creates a flywheel quadratic optimization problem with Sleipnir.
 *
 * @param dt Timestep duration.
 * @param N Number of samples in the problem.
 */
sleipnir::OptimizationProblem FlywheelSleipnir(std::chrono::duration<double> dt,
                                               int N);

/**
 * Creates a cart-pole nonlinear optimization problem with Sleipnir.
 *
 * @param dt Timestep duration.
 * @param N Number of samples in the problem.
 */
sleipnir::OptimizationProblem CartPoleSleipnir(std::chrono::duration<double> dt,
                                               int N);
