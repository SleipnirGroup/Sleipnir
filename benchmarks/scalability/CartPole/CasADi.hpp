// Copyright (c) Joshua Nichols and Tyler Veness

#pragma once

#include <casadi/casadi.hpp>
#include <units/time.h>

/**
 * Creates a cart-pole nonlinear optimization problem with CasADi.
 *
 * @param dt Timestep duration.
 * @param N Number of samples in the problem.
 * @param diagnostics True if diagnostic prints should be enabled.
 */
casadi::Opti CartPoleCasADi(units::second_t dt, int N);
