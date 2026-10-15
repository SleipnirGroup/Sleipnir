// Copyright (c) Sleipnir contributors

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include <variant>

namespace nb = nanobind;

namespace slp {

/// A 2D index of the form (row, col) where each element is an int or a slice.
///
/// The element types only affect the generated type stubs; the tuple's contents
/// are checked at runtime.
using SliceTuple = nb::typed<nb::tuple, std::variant<int, nb::slice>,
                             std::variant<int, nb::slice>>;

}  // namespace slp
