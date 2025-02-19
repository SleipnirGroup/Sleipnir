// Copyright (c) Sleipnir contributors

#pragma once

#include <chrono>

#include <sleipnir/optimization/optimization_problem.hpp>

/**
 * Creates a cart-pole nonlinear optimization problem with Sleipnir.
 *
 * @param dt Timestep duration.
 * @param N Number of samples in the problem.
 */
sleipnir::OptimizationProblem cart_pole_sleipnir(
    std::chrono::duration<double> dt, int N);
