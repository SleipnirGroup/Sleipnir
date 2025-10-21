// Copyright (c) Sleipnir contributors

#pragma once

#include <sleipnir/optimization/ocp.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "explicit_double.hpp"

extern template class slp::OCP<ExplicitDouble>;
extern template class slp::Problem<ExplicitDouble>;

#define SCALAR_TYPES_UNDER_TEST double, ExplicitDouble
