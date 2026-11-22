// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/solver/sqp.hpp"

#include <functional>
#include <span>

#include <Eigen/Core>

#include "sleipnir/optimization/solver/control_flow.hpp"
#include "sleipnir/optimization/solver/exit_status.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/options.hpp"
#include "sleipnir/optimization/solver/sqp_matrix_callbacks.hpp"
#include "sleipnir/optimization/solver/util/iterates.hpp"
#include "sleipnir/util/symbol_exports.hpp"

template SLEIPNIR_DLLEXPORT slp::ExitStatus slp::sqp(
    const SQPMatrixCallbacks<double>& matrix_callbacks,
    std::span<std::function<ControlFlow(const IterationInfo<double>& info)>>
        iteration_callbacks,
    const slp::Options<double>& options, Eigen::VectorX<double>& x);

template SLEIPNIR_DLLEXPORT slp::ExitStatus slp::sqp(
    const SQPMatrixCallbacks<double>& matrix_callbacks,
    std::span<std::function<ControlFlow(const IterationInfo<double>& info)>>
        iteration_callbacks,
    const Options<double>& options, SQPIterates<double>& iterates);
