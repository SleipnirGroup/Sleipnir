// Copyright (c) Sleipnir contributors

#include <sleipnir/autodiff/hessian.hpp>

#include "explicit_double.hpp"

template class slp::Hessian<ExplicitDouble, Eigen::Lower | Eigen::Upper>;
