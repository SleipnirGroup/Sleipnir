// Copyright (c) Sleipnir contributors

#pragma once

#include <pybind11/eigen.h>

namespace py = pybind11;

namespace sleipnir {

/**
 * Returns true if the given function input is a NumPy array containing an
 * arithmetic type.
 */
inline bool IsNumPyArithmeticArray(const auto& input) {
  return py::isinstance<py::array_t<double>>(input) ||
         py::isinstance<py::array_t<int64_t>>(input) ||
         py::isinstance<py::array_t<int32_t>>(input);
}

}  // namespace sleipnir
