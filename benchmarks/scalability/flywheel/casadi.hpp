// Copyright (c) Sleipnir contributors

#pragma once

#include <chrono>

#include <casadi/casadi.hpp>

/**
 * Creates a flywheel quadratic optimization problem with CasADi.
 *
 * @param dt Timestep duration.
 * @param N Number of samples in the problem.
 */
casadi::Opti flywheel_casadi(std::chrono::duration<double> dt, int N);
