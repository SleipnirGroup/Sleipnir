// Copyright (c) Sleipnir contributors

#include <sleipnir/optimization/solver/sqp.hpp>

#include "explicit_double.hpp"

template slp::ExitStatus slp::sqp(
    const SQPMatrixCallbacks<ExplicitDouble>& matrix_callbacks,
    std::span<std::function<bool(const IterationInfo<ExplicitDouble>& info)>>
        iteration_callbacks,
    const slp::Options& options,
    Eigen::Vector<ExplicitDouble, Eigen::Dynamic>& x);
