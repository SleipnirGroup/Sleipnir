// Copyright (c) Sleipnir contributors

#pragma once

#include <system_error>
#include <utility>

#include <fmt/core.h>

namespace sleipnir {

/**
 * Wrapper around fmt::print() that squelches write failure exceptions.
 */
template <typename... T>
void print(fmt::format_string<T...> fmt, T&&... args) {
  try {
    fmt::print(fmt, std::forward<T>(args)...);
  } catch (const std::system_error&) {
  }
}

/**
 * Wrapper around fmt::print() that squelches write failure exceptions.
 */
template <typename... T>
void print(std::FILE* f, fmt::format_string<T...> fmt, T&&... args) {
  try {
    fmt::print(f, fmt, std::forward<T>(args)...);
  } catch (const std::system_error&) {
  }
}

}  // namespace sleipnir
