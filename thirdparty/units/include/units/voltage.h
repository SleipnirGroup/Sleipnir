// Copyright (c) Joshua Nichols and Tyler Veness

// Copyright (c) 2016 Nic Holthaus
//
// The MIT License (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "units/base.h"

namespace units {
/**
 * @namespace units::voltage
 * @brief namespace for unit types and containers representing voltage values
 * @details The SI unit for voltage is `volts`, and the corresponding
 *          `base_unit` category is `voltage_unit`.
 * @anchor voltageContainers
 * @sa See unit_t for more information on unit type containers.
 */
#if !defined(DISABLE_PREDEFINED_UNITS) || \
    defined(ENABLE_PREDEFINED_VOLTAGE_UNITS)
UNIT_ADD_WITH_METRIC_PREFIXES(
    voltage, volt, volts, V, unit<std::ratio<1>, units::category::voltage_unit>)
UNIT_ADD(voltage, statvolt, statvolts, statV,
         unit<std::ratio<1000000, 299792458>, volts>)
UNIT_ADD(voltage, abvolt, abvolts, abV, unit<std::ratio<1, 100000000>, volts>)

UNIT_ADD_CATEGORY_TRAIT(voltage)
#endif

using namespace voltage;
}  // namespace units