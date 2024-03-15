// Copyright (c) Sleipnir contributors

#include "sleipnir/autodiff/Expression.hpp"

namespace sleipnir {

// Instantiate Expression pool in Expression.cpp instead to avoid ODR violation
template EXPORT_TEMPLATE_DEFINE(SLEIPNIR_DLLEXPORT)
    PoolAllocator<detail::Expression> GlobalPoolAllocator<detail::Expression>();

}  // namespace sleipnir
