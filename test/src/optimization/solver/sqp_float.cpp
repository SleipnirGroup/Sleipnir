// Copyright (c) Sleipnir contributors

#include <functional>
#include <span>

#include <Eigen/Core>
#include <sleipnir/optimization/solver/control_flow.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>
#include <sleipnir/optimization/solver/iteration_info.hpp>
#include <sleipnir/optimization/solver/options.hpp>
#include <sleipnir/optimization/solver/sqp.hpp>  // IWYU pragma: keep
#include <sleipnir/optimization/solver/sqp_matrix_callbacks.hpp>
#include <sleipnir/optimization/solver/util/iterates.hpp>

template slp::ExitStatus slp::sqp(
    const SQPMatrixCallbacks<float>& matrix_callbacks,
    std::span<std::function<ControlFlow(const IterationInfo<float>& info)>>
        iteration_callbacks,
    const slp::Options<float>& options,
    Eigen::Vector<float, Eigen::Dynamic>& x);

template slp::ExitStatus slp::sqp(
    const SQPMatrixCallbacks<float>& matrix_callbacks,
    std::span<std::function<ControlFlow(const IterationInfo<float>& info)>>
        iteration_callbacks,
    const Options<float>& options, SQPIterates<float>& iterates);
