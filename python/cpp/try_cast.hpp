// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <optional>
#include <utility>

#include <Eigen/Core>
#include <nanobind/ndarray.h>
#include <sleipnir/util/assert.hpp>

namespace nb = nanobind;

namespace slp {

/// Converts the given nb::object to a C++ type.
template <typename T>
std::optional<T> try_cast(const nb::object& obj) {
  if (nb::isinstance<T>(obj)) {
    return nb::cast<T>(obj);
  } else {
    return std::nullopt;
  }
}

/// Converts the given nb::ndarray to an Eigen matrix.
///
/// @tparam Scalar The Eigen matrix's scalar type.
/// @param obj The nb::ndarray.
template <typename Scalar>
std::optional<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>
try_cast_to_eigen(const nb::object& obj) {
  if (nb::isinstance<nb::ndarray<Scalar>>(obj)) {
    auto arr = nb::cast<nb::ndarray<Scalar>>(obj);
    slp_assert(arr.ndim() == 2);

    using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>,
                      alignof(Scalar), Stride>{
        arr.data(), static_cast<Eigen::Index>(arr.shape(0)),
        static_cast<Eigen::Index>(arr.shape(1)),
        Stride{arr.stride(0), arr.stride(1)}};
  } else {
    return std::nullopt;
  }
}

namespace detail {

/// Converts the given nb::ndarray to an Eigen type with the first scalar type
/// that works, then calls a function on it and returns the result.
///
/// @tparam Scalar Scalar type to try.
/// @tparam Scalars Rest of Scalar types to try.
/// @tparam F Type of function to apply.
/// @param f Function to apply.
/// @param obj The nb::ndarray.
template <typename Scalar, typename... Scalars, typename F>
std::optional<nb::object> apply_eigen_op(F&& f, const nb::object& obj) {
  if (auto mat = try_cast_to_eigen<Scalar>(obj)) {
    return nb::cast(f(mat.value()));
  } else if constexpr (sizeof...(Scalars) > 0) {
    return apply_eigen_op<Scalars...>(f, obj);
  } else {
    return std::nullopt;
  }
}

}  // namespace detail

/// Converts the given nb::ndarray to an Eigen type, then calls a function on it
/// and returns the result.
///
/// @tparam F Type of function to apply.
/// @param f Function to apply.
/// @param obj The nb::ndarray.
template <typename F>
std::optional<nb::object> apply_eigen_op(F&& f, const nb::object& obj) {
  return detail::apply_eigen_op<double, float, int64_t, int32_t>(
      std::forward<F>(f), obj);
}

}  // namespace slp
