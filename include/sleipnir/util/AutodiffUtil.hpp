// Copyright (c) Sleipnir contributors

#pragma once

#include <vector>

#include <Eigen/Core>

#include "sleipnir/SymbolExports.hpp"
#include "sleipnir/autodiff/Variable.hpp"

namespace sleipnir {

/**
 * Assigns an Eigen::VectorXd to a std::vector<Variable>.
 *
 * @param[out] dest The std::vector<Variable>.
 * @param[in] src The Eigen::VectorXd.
 */
SLEIPNIR_DLLEXPORT void SetAD(std::vector<Variable>& dest,
                              const Eigen::Ref<const Eigen::VectorXd>& src);

/**
 * Assigns an Eigen::VectorXd to an Eigen::Vector<Variable>.
 *
 * @param[out] dest The Eigen::Vector<Variable>.
 * @param[in] src The Eigen::VectorXd.
 */
SLEIPNIR_DLLEXPORT void SetAD(Eigen::Ref<VectorXvar> dest,
                              const Eigen::Ref<const Eigen::VectorXd>& src);

/**
 * Returns a std::vector<Variable> as an Eigen::VectorXd.
 *
 * @param src The std::vector<Variable>.
 */
SLEIPNIR_DLLEXPORT Eigen::VectorXd GetAD(std::vector<Variable> src);

}  // namespace sleipnir
