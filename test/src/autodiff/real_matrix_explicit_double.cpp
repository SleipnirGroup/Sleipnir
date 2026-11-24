// Copyright (c) Sleipnir contributors

#include <sleipnir/autodiff/real_matrix.hpp>

#include "explicit_double.hpp"

template class slp::RealMatrix<ExplicitDouble>;

template slp::RealMatrix<ExplicitDouble> slp::solve(
    const slp::RealMatrix<ExplicitDouble>& A,
    const slp::RealMatrix<ExplicitDouble>& B);
