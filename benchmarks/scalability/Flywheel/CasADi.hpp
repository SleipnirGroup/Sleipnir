// Copyright (c) Joshua Nichols and Tyler Veness

#pragma once

#include <casadi/casadi.hpp>
#include <units/time.h>

/**
 * Creates a flywheel quadratic optimization problem with CasADi.
 *
 * @param dt Timestep duration.
 * @param N Number of samples in the problem.
 */
casadi::Opti FlywheelCasADi(units::second_t dt, int N);
