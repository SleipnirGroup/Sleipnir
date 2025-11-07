// Copyright (c) Sleipnir contributors

#include <sleipnir/autodiff/variable_matrix.hpp>

#include "explicit_double.hpp"

template slp::VariableMatrix<ExplicitDouble> slp::solve(
    const slp::VariableMatrix<ExplicitDouble>& A,
    const slp::VariableMatrix<ExplicitDouble>& B);
