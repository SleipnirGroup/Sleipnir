// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>
#include <sleipnir/autodiff/variable_matrix.hpp>

Eigen::Vector<double, 5> differential_drive_dynamics_double(
    const Eigen::Vector<double, 5>& x, const Eigen::Vector<double, 2>& u);

sleipnir::VariableMatrix differential_drive_dynamics(
    const sleipnir::VariableMatrix& x, const sleipnir::VariableMatrix& u);
