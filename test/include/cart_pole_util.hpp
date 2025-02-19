// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>
#include <sleipnir/autodiff/variable_matrix.hpp>

Eigen::Vector<double, 4> cart_pole_dynamics_double(
    const Eigen::Vector<double, 4>& x, const Eigen::Vector<double, 1>& u);

sleipnir::VariableMatrix cart_pole_dynamics(const sleipnir::VariableMatrix& x,
                                            const sleipnir::VariableMatrix& u);
