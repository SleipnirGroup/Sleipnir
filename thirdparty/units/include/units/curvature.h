// Copyright (c) Sleipnir contributors

#pragma once

#include "units/angle.h"
#include "units/base.h"
#include "units/length.h"

namespace units {
using curvature_t = units::unit_t<
    units::compound_unit<units::radians, units::inverse<units::meters>>>;
}  // namespace units
