// Copyright (c) Sleipnir contributors

#pragma once

namespace slp::detail {

/// Type tag used to designate an uninitialized Matrix.
struct empty_t {};

/// Designates an uninitialized Matrix.
static constexpr empty_t empty{};

}  // namespace slp::detail
