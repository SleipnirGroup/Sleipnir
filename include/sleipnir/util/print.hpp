// Copyright (c) Sleipnir contributors

#pragma once

#include <cstdio>
#include <print>
#include <system_error>
#include <utility>

namespace sleipnir {

/**
 * Wrapper around std::print() that squelches write failure exceptions.
 */
template <typename... T>
inline void print(std::format_string<T...> fmt, T&&... args) {
  try {
    std::print(fmt, std::forward<T>(args)...);
  } catch (const std::system_error&) {
  }
}

/**
 * Wrapper around std::print() that squelches write failure exceptions.
 */
template <typename... T>
inline void print(std::FILE* f, std::format_string<T...> fmt, T&&... args) {
  try {
    std::print(f, fmt, std::forward<T>(args)...);
  } catch (const std::system_error&) {
  }
}

/**
 * Wrapper around std::println() that squelches write failure exceptions.
 */
template <typename... T>
inline void println(std::format_string<T...> fmt, T&&... args) {
  try {
    std::println(fmt, std::forward<T>(args)...);
  } catch (const std::system_error&) {
  }
}

/**
 * Wrapper around std::println() that squelches write failure exceptions.
 */
template <typename... T>
inline void println(std::FILE* f, std::format_string<T...> fmt, T&&... args) {
  try {
    std::println(f, fmt, std::forward<T>(args)...);
  } catch (const std::system_error&) {
  }
}

}  // namespace sleipnir
