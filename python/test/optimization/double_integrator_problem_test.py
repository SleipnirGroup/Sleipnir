import math

import numpy as np
import numpy.typing as npt
import pytest
from scipy.signal import cont2discrete
from sleipnir.autodiff import ExpressionType
from sleipnir.optimization import ExitStatus, Problem, bounds


def optimal_double_integrator_inputs(
    r: float, dt: float, N: int
) -> npt.NDArray[np.float64]:
    """
    Returns the input trajectory that minimizes Σ(r − pₖ)² for k ∈ [0, N]
    subject to double integrator kinematics with timestep dt, starting and
    ending at rest at 0 and r respectively, |vₖ| ≤ 1, and |aₖ| ≤ 1.

    Parameter ``r``:
        Reference position.

    Parameter ``dt``:
        Timestep.

    Parameter ``N``:
        Number of timesteps.
    """
    v_max = 1.0
    a_max = 1.0
    t_n = N * dt

    # The optimal input accelerates to the velocity limit, cruises, then
    # decelerates past r and accelerates back to rest at r. In continuous time,
    # let vₘₐₓ = 1 and aₘₐₓ = 1 be the velocity and acceleration limits and
    # tₙ = N dt be the total time. A phase with input u and duration t changes
    # the velocity by ut, and if it starts at velocity v₀, it covers
    # v₀t + ut²/2. The phases are
    #
    #   Phase      │ Duration │ u     │ Final v │ Distance
    #   ───────────┼──────────┼───────┼─────────┼───────────────────
    #   Accelerate │ t₁       │ aₘₐₓ  │ vₘₐₓ    │ aₘₐₓt₁²/2
    #   Cruise     │ t₂       │ 0     │ vₘₐₓ    │ vₘₐₓt₂
    #   Decelerate │ t₃       │ −aₘₐₓ │ −aₘₐₓt₄ │ vₘₐₓt₃ − aₘₐₓt₃²/2
    #   Accelerate │ t₄       │ aₘₐₓ  │ 0       │ −aₘₐₓt₄²/2
    #
    # where t₁ = vₘₐₓ/aₘₐₓ. The phases fill the total time. Deceleration
    # starts at vₘₐₓ and the final acceleration ends at rest, so the velocity
    # between them is both vₘₐₓ − aₘₐₓt₃ and −aₘₐₓt₄. Therefore,
    #
    #   t₁ + t₂ + t₃ + t₄ = tₙ
    #   vₘₐₓ − aₘₐₓt₃ = −aₘₐₓt₄
    #
    # Dividing the second equation by aₘₐₓ gives t₄ = t₃ − t₁, and substituting
    # that into the first gives t₂ = tₙ − 2t₃. Ending at r requires the
    # distances to sum to r. Dividing that sum by aₘₐₓ and substituting
    # vₘₐₓ/aₘₐₓ = t₁ gives
    #
    #   t₁²/2 + t₁t₂ + t₁t₃ − (t₃² + t₄²)/2 = r/aₘₐₓ
    #   t₁²/2 + t₁(tₙ − 2t₃) + t₁t₃ − (t₃² + (t₃ − t₁)²)/2 = r/aₘₐₓ
    #   t₁²/2 + t₁tₙ − t₁t₃ − t₃² + t₁t₃ − t₁²/2 = r/aₘₐₓ
    #   t₁tₙ − t₃² = r/aₘₐₓ
    #   t₃ = √(t₁tₙ − r/aₘₐₓ)
    #
    # so the phase durations are
    #
    #   t₁ = vₘₐₓ/aₘₐₓ
    #   t₂ = tₙ − 2t₃
    #   t₃ = √(t₁tₙ − r/aₘₐₓ)
    #   t₄ = t₃ − t₁
    t_1 = v_max / a_max
    t_3 = math.sqrt(t_1 * t_n - r / a_max)
    t_2 = t_n - 2.0 * t_3
    t_4 = t_3 - t_1

    # Each phase starts when the previous one ends, and the last ends at tₙ, so
    # the phases start at 0, t₁, t₁ + t₂, and tₙ − t₄.
    #
    # The kinematics are exact for piecewise constant acceleration, so the
    # discrete optimum switches at the same times. Timestep k applies uₖ over
    # [k dt, (k + 1) dt), so cruising, deceleration, and the final acceleration
    # start in timesteps
    #
    #   k₂ = t₁/dt
    #   k₃ = ⌊(t₁ + t₂)/dt⌋
    #   k₄ = ⌊(tₙ − t₄)/dt⌋
    #
    # t₁/dt is assumed to be an integer, so cruising starts on a timestep
    # boundary.
    k_2 = round(t_1 / dt)
    k_3 = math.floor((t_1 + t_2) / dt)
    k_4 = math.floor((t_n - t_4) / dt)

    # Every timestep's input is its phase's u, except in timesteps k₃ and k₄.
    # Deceleration and the final acceleration generally start partway through
    # those timesteps, so they take fractional inputs, which start at zero and
    # are solved for below.
    U = np.zeros(N)
    U[:k_2] = a_max
    U[k_3 + 1 : k_4] = -a_max
    U[k_4 + 1 :] = a_max

    # Continuous dynamics ẋ = Ax + Bu with state x = [p  v]ᵀ and input u = a
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    A_d, B_d, *_ = cont2discrete((A, B, np.eye(2), np.zeros((2, 1))), dt)

    def final_state(inputs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Returns the state at timestep N reached from rest at 0 with the inputs.
        """
        x = np.zeros((2, 1))
        for u in inputs:
            x = A_d @ x + B_d * u
        return x

    def unit(k: int) -> npt.NDArray[np.float64]:
        """
        Returns the inputs that are zero except for a unit input at timestep k.
        """
        e = np.zeros(N)
        e[k] = 1.0
        return e

    # The discrete dynamics xₖ₊₁ = A_d xₖ + B_d uₖ are exact for piecewise
    # constant inputs and linear in the inputs, so the final state is
    #
    #   xₙ = x_fixed + c₃uₖ₃ + c₄uₖ₄
    #
    # where x_fixed is the final state from the fixed inputs alone, and c₃ and
    # c₄ are the final states from unit inputs at timesteps k₃ and k₄. The
    # fractional inputs are still zero in U, so x_fixed is the final state
    # from U. Ending at rest at r requires
    #
    #           [uₖ₃]   [r]
    #   [c₃  c₄][uₖ₄] = [0] − x_fixed
    #
    # which is a 2×2 linear system for the fractional inputs.
    x_fixed = final_state(U)
    C = np.hstack([final_state(unit(k_3)), final_state(unit(k_4))])
    U[k_3], U[k_4] = np.linalg.solve(C, np.array([[r], [0.0]]) - x_fixed)[:, 0]

    return U


def test_double_integrator_problem() -> None:
    TOTAL_TIME = 3.5
    dt = 0.005
    N = int(TOTAL_TIME / dt)

    r = 2.0

    problem = Problem()

    # 2x1 state vector with N + 1 timesteps (includes last state)
    X = problem.decision_variable(2, N + 1)

    # 1x1 input vector with N timesteps (input at last state doesn't matter)
    U = problem.decision_variable(1, N)

    # Kinematics constraint assuming constant acceleration between timesteps
    for k in range(N):
        t = dt
        p_k1 = X[0, k + 1]
        v_k1 = X[1, k + 1]
        p_k = X[0, k]
        v_k = X[1, k]
        a_k = U[0, k]

        # pₖ₊₁ = pₖ + vₖt + 1/2aₖt²
        problem.subject_to(p_k1 == p_k + v_k * t + 0.5 * a_k * t**2)

        # vₖ₊₁ = vₖ + aₖt
        problem.subject_to(v_k1 == v_k + a_k * t)

    # Start and end at rest
    problem.subject_to(X[:, :1] == np.array([[0.0], [0.0]]))
    problem.subject_to(X[:, N : N + 1] == np.array([[r], [0.0]]))

    # Limit velocity
    problem.subject_to(bounds(-1, X[1:2, :], 1))

    # Limit acceleration
    problem.subject_to(bounds(-1, U, 1))

    # Cost function - minimize position error
    problem.minimize(sum((r - X[0, k]) ** 2 for k in range(N + 1)))

    assert problem.cost_function_type() == ExpressionType.QUADRATIC
    assert problem.equality_constraint_type() == ExpressionType.LINEAR
    assert problem.inequality_constraint_type() == ExpressionType.LINEAR

    assert problem.solve(diagnostics=True) == ExitStatus.SUCCESS

    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    A_d, B_d, *_ = cont2discrete((A, B, np.eye(2), np.zeros((2, 1))), dt, method="zoh")

    U_ref = optimal_double_integrator_inputs(r, dt, N)

    # Only position is in the cost, and vₖ affects it only through pₖ₊₁, so
    # velocity is determined less precisely than position. uₖ affects only vₖ₊₁
    # through the dt uₖ it adds, so U is determined 1/dt times less precisely
    # than velocity.
    p_tolerance = 1e-4
    v_tolerance = 1e-3
    u_tolerance = v_tolerance / dt

    # Verify initial state
    assert X.value(0, 0) == pytest.approx(0.0, abs=1e-8)
    assert X.value(1, 0) == pytest.approx(0.0, abs=1e-8)

    # Verify solution
    x_ref = np.zeros((2, 1))
    for k in range(N + 1):
        assert X.value(0, k) == pytest.approx(x_ref[0, 0], abs=p_tolerance)
        assert X.value(1, k) == pytest.approx(x_ref[1, 0], abs=v_tolerance)

        if k < N:
            assert U.value(0, k) == pytest.approx(U_ref[k], abs=u_tolerance)
            x_ref = A_d @ x_ref + B_d * U_ref[k]

    # Verify final state
    assert X.value(0, N) == pytest.approx(r, abs=1e-8)
    assert X.value(1, N) == pytest.approx(0.0, abs=1e-8)

    # Log states for offline viewing
    with open("Double integrator states.csv", "w") as f:
        f.write("Time (s),Position (m),Velocity (rad/s)\n")

        for k in range(N + 1):
            f.write(f"{k * dt},{X.value(0, k)},{X.value(1, k)}\n")

    # Log inputs for offline viewing
    with open("Double integrator inputs.csv", "w") as f:
        f.write("Time (s),Acceleration (m/s²)\n")

        for k in range(N + 1):
            if k < N:
                f.write(f"{k * dt},{U.value(0, k)}\n")
            else:
                f.write(f"{k * dt},0.0\n")
