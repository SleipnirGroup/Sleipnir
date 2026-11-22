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

#include "explicit_double.hpp"

template slp::ExitStatus slp::sqp(
    const SQPMatrixCallbacks<ExplicitDouble>& matrix_callbacks,
    std::span<
        std::function<ControlFlow(const IterationInfo<ExplicitDouble>& info)>>
        iteration_callbacks,
    const slp::Options<ExplicitDouble>& options,
    Eigen::VectorX<ExplicitDouble>& x);

template slp::ExitStatus slp::sqp(
    const SQPMatrixCallbacks<ExplicitDouble>& matrix_callbacks,
    std::span<
        std::function<ControlFlow(const IterationInfo<ExplicitDouble>& info)>>
        iteration_callbacks,
    const Options<ExplicitDouble>& options,
    SQPIterates<ExplicitDouble>& iterates);
