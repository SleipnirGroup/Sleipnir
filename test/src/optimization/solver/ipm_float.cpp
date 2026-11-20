// Copyright (c) Sleipnir contributors

#include <functional>
#include <span>

#include <Eigen/Core>
#include <sleipnir/optimization/solver/control_flow.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>
#include <sleipnir/optimization/solver/ipm.hpp>  // IWYU pragma: keep
#include <sleipnir/optimization/solver/ipm_matrix_callbacks.hpp>
#include <sleipnir/optimization/solver/iteration_info.hpp>
#include <sleipnir/optimization/solver/options.hpp>

template slp::ExitStatus slp::ipm(
    const IPMMatrixCallbacks<float>& matrix_callbacks,
    std::span<std::function<ControlFlow(const IterationInfo<float>& info)>>
        iteration_callbacks,
    const Options<float>& options,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Eigen::Vector<float, Eigen::Dynamic>& x);

template slp::ExitStatus slp::ipm(
    const IPMMatrixCallbacks<float>& matrix_callbacks,
    std::span<std::function<ControlFlow(const IterationInfo<float>& info)>>
        iteration_callbacks,
    const Options<float>& options, bool in_feasibility_restoration,
#ifdef SLEIPNIR_ENABLE_BOUND_PROJECTION
    const Eigen::ArrayX<bool>& bound_constraint_mask,
#endif
    Eigen::VectorX<float>& x, Eigen::VectorX<float>& s,
    Eigen::VectorX<float>& y, Eigen::VectorX<float>& z, float& μ,
    int& iterations);
