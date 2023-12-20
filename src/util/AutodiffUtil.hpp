// Copyright (c) Sleipnir contributors

#pragma once

#include <Eigen/Core>

#include "sleipnir/autodiff/Variable.hpp"

namespace sleipnir {

/**
 * Assigns an Eigen::VectorXd to an Eigen::Vector<Variable>.
 *
 * @param[out] dest The Eigen::Vector<Variable>.
 * @param[in] src The Eigen::VectorXd.
 */
void SetAD(Eigen::Ref<VectorXvar> dest,
           const Eigen::Ref<const Eigen::VectorXd>& src);

/**
 * Returns an Eigen::Vector<Variable> as an Eigen::VectorXd.
 *
 * @param src The Eigen::Vector<Variable>.
 */
Eigen::VectorXd GetAD(Eigen::Ref<VectorXvar> src);

}  // namespace sleipnir
