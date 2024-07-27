// Copyright (c) Sleipnir contributors

#pragma once

#include <optional>

#include <Eigen/Core>
#include <nanobind/ndarray.h>
#include <sleipnir/util/Assert.hpp>

namespace nb = nanobind;

namespace sleipnir {

/**
 * Converts the given nb::object to a C++ type.
 */
template <typename T>
inline std::optional<T> TryCast(const nb::object& obj) {
  if (nb::isinstance<T>(obj)) {
    return nb::cast<T>(obj);
  } else {
    return std::nullopt;
  }
}

/**
 * Converts the given nb::ndarray to an Eigen matrix.
 */
template <typename T>
inline std::optional<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
TryCastToEigen(const nb::object& obj) {
  if (nb::isinstance<nb::ndarray<T>>(obj)) {
    auto input = nb::cast<nb::ndarray<T>>(obj);
    Assert(input.ndim() == 2);

    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                          Eigen::RowMajor>>{
        input.data(),
        static_cast<Eigen::Index>(input.shape(0)),
        static_cast<Eigen::Index>(input.shape(1)),
        {input.stride(0), input.stride(1)}};
  } else {
    return std::nullopt;
  }
}

}  // namespace sleipnir
