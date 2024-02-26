// Copyright (c) Sleipnir contributors

#pragma once

#include <chrono>

#include <casadi/casadi.hpp>

/**
 * Creates a cart-pole nonlinear optimization problem with CasADi.
 *
 * @param dt Timestep duration.
 * @param N Number of samples in the problem.
 * @param diagnostics True if diagnostic prints should be enabled.
 */
casadi::Opti CartPoleCasADi(std::chrono::duration<double> dt, int N);
