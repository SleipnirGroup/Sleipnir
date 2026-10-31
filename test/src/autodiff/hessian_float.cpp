// Copyright (c) Sleipnir contributors

#include <Eigen/Core>
#include <sleipnir/autodiff/hessian.hpp>

template class slp::Hessian<float, Eigen::Lower | Eigen::Upper>;
