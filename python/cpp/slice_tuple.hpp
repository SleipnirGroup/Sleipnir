// Copyright (c) Sleipnir contributors

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include <tuple>
#include <variant>

#include <sleipnir/autodiff/slice.hpp>

#include "try_cast.hpp"

namespace nb = nanobind;

namespace slp {

/// A 2D index of the form (row, col) where each element is an int or a slice.
///
/// The element types only affect the generated type stubs; the tuple's contents
/// are checked at runtime.
using SliceTuple = nb::typed<nb::tuple, std::variant<int, nb::slice>,
                             std::variant<int, nb::slice>>;

/// Converts a Python index or slice into a Slice and its length.
///
/// @param elem The Python index or slice.
/// @param length The length of the sliced dimension.
/// @return The Slice and its length.
inline std::tuple<Slice, int> to_slice(const nb::object& elem, int length) {
  if (auto py_slice = try_cast<nb::slice>(elem)) {
    auto t = py_slice.value().compute(length);
    return {Slice{t.get<0>(), t.get<1>(), t.get<2>()},
            static_cast<int>(t.get<3>())};
  } else {
    int start = nb::cast<int>(elem);
    if (start < 0) {
      start += length;
    }
    if (start < 0 || start >= length) {
      throw nb::index_error("Index out of bounds");
    }
    return {Slice{start, start + 1}, 1};
  }
}

}  // namespace slp
