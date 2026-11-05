import math
from collections.abc import Callable

import numpy as np
import pytest
from flywheel_util import optimal_flywheel_trajectory
from sleipnir.autodiff import ExpressionType, VariableMatrix
from sleipnir.optimization import (
    OCP,
    DynamicsType,
    ExitStatus,
    TimestepMethod,
    TranscriptionMethod,
)
from tolerance import SolutionType, solution_tolerance


def flywheel_test(
    test_name: str,
    A: float,
    B: float,
    f: Callable[[VariableMatrix, VariableMatrix], VariableMatrix],
    dynamics_type: DynamicsType,
    transcription_method: TranscriptionMethod,
) -> None:
    TOTAL_TIME = 5.0  # s
    dt = 0.005  # s
    N = int(TOTAL_TIME / dt)

    # Flywheel model:
    # States: [velocity]
    # Inputs: [voltage]
    A_discrete = math.exp(A * dt)
    B_discrete = (1.0 - A_discrete) * B

    r = 10.0

    problem = OCP(
        1, 1, dt, N, f, dynamics_type, TimestepMethod.FIXED, transcription_method
    )
    problem.constrain_initial_state(0.0)
    problem.set_upper_input_bound(12)
    problem.set_lower_input_bound(-12)

    # Set up cost
    r_mat = np.full((1, N + 1), r)
    problem.minimize((r_mat - problem.X()) @ (r_mat - problem.X()).T)

    assert problem.cost_function_type() == ExpressionType.QUADRATIC
    if transcription_method == TranscriptionMethod.SINGLE_SHOOTING:
        # The initial state is set directly instead of constrained
        assert problem.equality_constraint_type() == ExpressionType.NONE
    else:
        assert problem.equality_constraint_type() == ExpressionType.LINEAR
    assert problem.inequality_constraint_type() == ExpressionType.LINEAR

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    X_ref, U_ref = optimal_flywheel_trajectory(A_discrete, B_discrete, r, 12.0, N)

    # Verify initial state
    assert problem.X().value(0, 0) == pytest.approx(0.0, abs=solution_tolerance(1))

    # Verify solution
    for k in range(N + 1):
        if transcription_method == TranscriptionMethod.DIRECT_COLLOCATION:
            # After the input leaves saturation, the spline's inputs chatter (see
            # below). The cost barely depends on that alternating mode, so the
            # solution is degenerate.
            assert problem.X().value(0, k) == pytest.approx(
                X_ref[k], abs=solution_tolerance(5, SolutionType.DEGENERATE)
            )
            if k < N:
                # The tolerance is large because the trajectory is represented by a
                # spline, and splines chatter when transitioning quickly between
                # steady-states.
                assert problem.U().value(0, k) == pytest.approx(U_ref[k], abs=20.0)
        else:
            assert problem.X().value(0, k) == pytest.approx(
                X_ref[k], abs=solution_tolerance(1000)
            )
            if k < N:
                # The cost only depends on uₖ through the Buₖ it adds to xₖ₊₁,
                # so compare Buₖ, which is determined as precisely as xₖ₊₁
                assert B_discrete * problem.U().value(0, k) == pytest.approx(
                    B_discrete * U_ref[k], abs=solution_tolerance(1000)
                )

    # Log states for offline viewing
    with open(f"{test_name} states.csv", "w") as file:
        file.write("Time (s),Velocity (rad/s)\n")

        for k in range(N + 1):
            file.write(f"{k * dt},{problem.X().value(0, k)}\n")

    # Log inputs for offline viewing
    with open(f"{test_name} inputs.csv", "w") as file:
        file.write("Time (s),Voltage (V)\n")

        for k in range(N + 1):
            if k < N:
                file.write(f"{k * dt},{problem.U().value(0, k)}\n")
            else:
                file.write(f"{k * dt},0.0\n")


def test_flywheel_explicit_ocp() -> None:
    A = -1.0
    B = 1.0

    def f_ode(x: VariableMatrix, u: VariableMatrix) -> VariableMatrix:
        return A * x + B * u

    flywheel_test(
        "OCP Flywheel Explicit Collocation",
        A,
        B,
        f_ode,
        DynamicsType.EXPLICIT_ODE,
        TranscriptionMethod.DIRECT_COLLOCATION,
    )
    flywheel_test(
        "OCP Flywheel Explicit Transcription",
        A,
        B,
        f_ode,
        DynamicsType.EXPLICIT_ODE,
        TranscriptionMethod.DIRECT_TRANSCRIPTION,
    )
    flywheel_test(
        "OCP Flywheel Explicit Single-Shooting",
        A,
        B,
        f_ode,
        DynamicsType.EXPLICIT_ODE,
        TranscriptionMethod.SINGLE_SHOOTING,
    )


def test_flywheel_discrete_ocp() -> None:
    A = -1.0
    B = 1.0
    dt = 0.005  # s

    A_discrete = math.exp(A * dt)
    B_discrete = (1.0 - A_discrete) * B

    def f_discrete(x: VariableMatrix, u: VariableMatrix) -> VariableMatrix:
        return A_discrete * x + B_discrete * u

    flywheel_test(
        "OCP Flywheel Discrete Transcription",
        A,
        B,
        f_discrete,
        DynamicsType.DISCRETE,
        TranscriptionMethod.DIRECT_TRANSCRIPTION,
    )
    flywheel_test(
        "OCP Flywheel Discrete Single-Shooting",
        A,
        B,
        f_discrete,
        DynamicsType.DISCRETE,
        TranscriptionMethod.SINGLE_SHOOTING,
    )
