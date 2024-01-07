// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>
#include <sleipnir/autodiff/VariableMatrix.hpp>

Eigen::Vector<double, 5> DifferentialDriveDynamicsDouble(
    const Eigen::Vector<double, 5>& x, const Eigen::Vector<double, 2>& u);

sleipnir::VariableMatrix DifferentialDriveDynamics(
    const sleipnir::VariableMatrix& x, const sleipnir::VariableMatrix& u);
