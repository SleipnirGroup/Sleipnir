// Copyright (c) Sleipnir contributors

#pragma once

namespace slp::detail {

/// Type tag used to designate an uninitialized RealMatrix.
struct empty_t {};

/// Designates an uninitialized RealMatrix.
static constexpr empty_t empty{};

}  // namespace slp::detail
