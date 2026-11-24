// Copyright (c) Sleipnir contributors

#include <sleipnir/autodiff/matrix.hpp>

#include "explicit_double.hpp"

template class slp::Matrix<ExplicitDouble>;

template slp::Matrix<ExplicitDouble> slp::solve(
    const slp::Matrix<ExplicitDouble>& A, const slp::Matrix<ExplicitDouble>& B);
