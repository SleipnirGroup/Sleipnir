// Copyright (c) Sleipnir contributors

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace sleipnir {

void BindVariableMatrices(py::module_& autodiff);

}  // namespace sleipnir
