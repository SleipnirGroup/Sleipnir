// Copyright (c) Joshua Nichols and Tyler Veness

#pragma once

#include <sleipnir/optimization/OptimizationProblem.h>
#include <units/time.h>

/**
 * Creates a flywheel quadratic optimization problem with OptimizationProblem.
 *
 * @param dt Timestep duration.
 * @param N Number of samples in the problem.
 */
sleipnir::OptimizationProblem FlywheelOptimizationProblem(units::second_t dt,
                                                          int N);

/**
 * Creates a cart-pole nonlinear optimization problem with OptimizationProblem.
 *
 * @param dt Timestep duration.
 * @param N Number of samples in the problem.
 */
sleipnir::OptimizationProblem CartPoleOptimizationProblem(units::second_t dt,
                                                          int N);
