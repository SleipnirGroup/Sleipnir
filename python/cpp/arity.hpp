// Copyright (c) Sleipnir contributors

#pragma once

#include <cstddef>
#include <type_traits>

namespace slp {

/// Type implicitly convertible to any type. Used for counting aggregate
/// members.
struct AnyType {
  template <typename T>
  // NOLINTNEXTLINE (google-explicit-constructor)
  operator T() const;
};

/// Returns the number of members in aggregate type T.
///
/// @tparam T Aggregate type.
/// @tparam Args Initializers for the members counted so far.
template <typename T, typename... Args>
  requires std::is_aggregate_v<T>
consteval size_t member_count() {
  if constexpr (requires { T{Args{}..., AnyType{}}; }) {
    return member_count<T, Args..., AnyType>();
  } else {
    return sizeof...(Args);
  }
}

/// Returns the number of parameters of a lambda.
///
/// @tparam F Lambda type.
template <typename F>
consteval size_t lambda_param_count() {
  return []<typename R, typename C, typename... Args>(R (C::*)(Args...) const) {
    return sizeof...(Args);
  }(&F::operator());
}

}  // namespace slp
