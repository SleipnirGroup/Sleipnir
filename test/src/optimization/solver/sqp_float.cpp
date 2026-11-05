// Copyright (c) Sleipnir contributors

#include <sleipnir/optimization/solver/sqp.hpp>

template slp::ExitStatus slp::sqp(
    const SQPMatrixCallbacks<float>& matrix_callbacks,
    std::span<std::function<bool(const IterationInfo<float>& info)>>
        iteration_callbacks,
    const slp::Options<float>& options,
    Eigen::Vector<float, Eigen::Dynamic>& x);
