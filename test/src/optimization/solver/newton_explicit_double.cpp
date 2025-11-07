// Copyright (c) Sleipnir contributors

#include <sleipnir/optimization/solver/newton.hpp>

#include "explicit_double.hpp"

template slp::ExitStatus slp::newton(
    const NewtonMatrixCallbacks<ExplicitDouble>& matrix_callbacks,
    std::span<std::function<bool(const IterationInfo<ExplicitDouble>& info)>>
        iteration_callbacks,
    const Options& options, Eigen::Vector<ExplicitDouble, Eigen::Dynamic>& x);
