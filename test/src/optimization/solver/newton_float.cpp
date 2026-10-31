// Copyright (c) Sleipnir contributors

#include <sleipnir/optimization/solver/newton.hpp>

template slp::ExitStatus slp::newton(
    const NewtonMatrixCallbacks<float>& matrix_callbacks,
    std::span<std::function<bool(const IterationInfo<float>& info)>>
        iteration_callbacks,
    const Options<float>& options, Eigen::Vector<float, Eigen::Dynamic>& x);
