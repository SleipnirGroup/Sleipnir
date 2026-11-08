#!/usr/bin/python

"""
Solve the guided fuel-optimal landing diversion (G-FOLD) problem.

The coordinate system is +X up, +Y east, +Z north.

[1] Açıkmeşe et al., "Lossless Convexification of Nonconvex Control Bound and
    Pointing Constraints of the Soft Landing Optimal Control Problem", 2013.
    http://www.larsblackmore.com/iee_tcst13.pdf
[2] Açıkmeşe et al., "Convex Programming Approach to Powered Descent Guidance
    for Mars Landing", 2007. https://sci-hub.st/10.2514/1.27553
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numpy.linalg import norm
from scipy.signal import cont2discrete
from sleipnir.autodiff import cos, sin
from sleipnir.optimization import ExitStatus, Problem, bounds


def lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def time_horizon_upper_bound(
    q_0: npt.NDArray[np.float64],
    v_0: npt.NDArray[np.float64],
    q_f: npt.NDArray[np.float64],
    v_f: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    ω: npt.NDArray[np.float64],
    Δv_max: float,
) -> float:
    """
    Returns the time horizon upper bound t_max (s) for a flight from (q₀, v₀)
    to (q_f, v_f) in a planet frame rotating at ω. Any longer flight would need
    a larger velocity change than the thrust can provide.

    See README.md for the derivation.

    Args:
        q_0: Initial position (m).
        v_0: Initial velocity (m/s).
        q_f: Final position (m).
        v_f: Final velocity (m/s).
        g: Gravitational acceleration (m/s²).
        ω: Angular velocity of planet (rad/s). Must be nonzero.
        Δv_max: Maximum velocity change the thrust can provide (m/s).

    Returns:
        The time horizon upper bound (s).
    """
    #        [ 0  −ω₃  ω₂]
    # S(ω) = [ ω₃  0  −ω₁]
    #        [−ω₂  ω₁  0 ]
    #
    # See equation (2) of [1].
    ω_1 = ω[0, 0]
    ω_2 = ω[1, 0]
    ω_3 = ω[2, 0]
    S = np.array([[0.0, -ω_3, ω_2], [ω_3, 0.0, -ω_1], [-ω_2, ω_1, 0.0]])

    # Integrating the dynamics in an inertial frame removes the planet rotation
    # terms and gives |Δv_u(t_f)| ≤ Δv_max where
    #
    #   Δv_u(t) = R(t)v_I,f − v_I,0 − R_int(t)g
    #
    # and v_I = v + ω×q is the inertial velocity in the planet frame.
    #
    # t_max is the largest t_f that satisfies the inequality, so we maximize t_f
    # subject to it.
    #
    # See equation (11) of README.md.
    problem = Problem()
    t_f = problem.decision_variable()

    # R(t) = I + k₁(t)S(ω) + k₂(t)S(ω)²
    # R_int(t) = tI + k₂(t)S(ω) + k₃(t)S(ω)²
    #
    # See equations (8) and (9) of README.md.
    ω_norm = norm(ω)
    k_1 = sin(ω_norm * t_f) / ω_norm
    k_2 = (1 - cos(ω_norm * t_f)) / ω_norm**2
    k_3 = (t_f - k_1) / ω_norm**2

    # See equation (5) of README.md.
    v_I_0 = v_0 + S @ q_0
    v_I_f = v_f + S @ q_f

    # Δv_u(t_f) = R(t_f)v_I,f − v_I,0 − R_int(t_f)g
    #           = v_I,f − v_I,0 − t_f g
    #             + k₁(t_f)S(ω)v_I,f
    #             + k₂(t_f)(S(ω)²v_I,f − S(ω)g)
    #             − k₃(t_f)S(ω)²g
    #
    # See equation (10) of README.md.
    Δv_u = (
        v_I_f
        - v_I_0
        - t_f * g
        + k_1 * (S @ v_I_f)
        + k_2 * (S @ S @ v_I_f - S @ g)
        - k_3 * (S @ S @ g)
    )

    problem.maximize(t_f)
    problem.subject_to(Δv_u.T @ Δv_u <= Δv_max**2)
    problem.solve()

    return t_f.value()


def time_horizon_lower_bound(
    q_0: npt.NDArray[np.float64],
    v_0: npt.NDArray[np.float64],
    q_f: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    ω: npt.NDArray[np.float64],
    v_max: float,
    a_max: float,
    θ: float,
    t_max: float,
) -> float:
    """
    Returns the time horizon lower bound t_min (s) for a flight from (q₀, v₀)
    to q_f with zero final vertical velocity in a planet frame rotating at ω.
    +X is up. Any shorter flight would need more vertical acceleration than
    gravity and the thrust can provide.

    See README.md for the derivation.

    Args:
        q_0: Initial position (m).
        v_0: Initial velocity (m/s).
        q_f: Final position (m).
        g: Gravitational acceleration (m/s²). Must point down.
        ω: Angular velocity of planet (rad/s).
        v_max: Maximum velocity magnitude (m/s).
        a_max: Maximum thrust acceleration (m/s²).
        θ: Thrust pointing limit from vertical (rad).
        t_max: Time horizon upper bound (s).

    Returns:
        The time horizon lower bound (s).
    """
    # The planet rotation terms add at most a_ω of vertical acceleration in
    # either direction.
    #
    # See equations (24) and (25) of README.md.
    q_bound = (norm(q_0) + norm(q_f) + v_max * t_max) / 2
    a_ω = norm(np.cross([1.0, 0.0, 0.0], ω[:, 0])) * (2 * v_max + norm(ω) * q_bound)

    # The thrust pointing limit keeps the vertical thrust acceleration within
    # [min(0, cos(θ))a_max, a_max], so the vertical acceleration lies within
    # [−a_fall, a_brake].
    #
    # See equations (22), (26), and (27) of README.md.
    a_fall = -g[0, 0] - min(0.0, math.cos(θ)) * a_max + a_ω
    a_brake = a_max + g[0, 0] + a_ω

    # The fastest descent falls at a_fall up to a peak downward speed s_peak,
    # then brakes at a_brake.
    #
    # See equations (28) and (29) of README.md.
    Δh = q_0[0, 0] - q_f[0, 0]
    s_0 = -v_0[0, 0]
    s_peak = math.sqrt((2 * Δh + s_0**2 / a_fall) / (1 / a_fall + 1 / a_brake))
    return (s_peak - s_0) / a_fall + s_peak / a_brake


def line_search(f: Callable[[int], Any], first: int, last: int) -> tuple[int, Any]:
    """
    Returns the index and value of the unimodal function f's minimum within [first, last].

    Args:
        f: Unimodal function.
        first: First index of range.
        last: Last index of range.

    Returns:
        Index and value of the unimodal function's minimum.
    """
    # https://en.wikipedia.org/wiki/Golden-section_search

    x_3 = first
    x_1 = last

    def evaluate(x: int) -> Any:
        f_x = f(x)
        print(f"\tN ∈ [{min(x_1, x_3)}, {max(x_1, x_3)}] -> N = {x}: {f_x}")
        return f_x

    if first + 1 == last:
        return min((first, evaluate(first)), (last, evaluate(last)), key=lambda p: p[1])

    ϕ_inv = (math.sqrt(5) - 1) / 2

    x_2 = round(lerp(x_3, x_1, ϕ_inv))
    f_2 = evaluate(x_2)

    while abs(x_3 - x_1) > 2:
        x_4 = round(lerp(x_3, x_2, ϕ_inv))
        if (f_4 := evaluate(x_4)) < f_2:
            # Keep [x_2, x_3]
            x_1 = x_2
            x_2 = x_4
            f_2 = f_4
        else:
            # Keep [x_1, x_4]
            x_3 = x_1
            x_1 = x_4

    # If x_1 or x_3 are first or last, they haven't been evaluated yet
    if x_1 in (first, last) and (f_1 := evaluate(x_1)) < f_2:
        return x_1, f_1
    if x_3 in (first, last) and (f_3 := evaluate(x_3)) < f_2:
        return x_3, f_3

    return x_2, f_2


@dataclass(eq=False)
class Solution:
    status: ExitStatus
    X_value: npt.NDArray[np.float64]
    Z_value: npt.NDArray[np.float64]
    U_value: npt.NDArray[np.float64]
    σ_value: npt.NDArray[np.float64]

    def cost(self) -> float:
        return self.σ_value.sum()

    def __lt__(self, other: "Solution") -> bool:
        return self._key() < other._key()

    def _key(self) -> tuple[bool, float]:
        return (self.status != ExitStatus.SUCCESS, self.cost())

    def __str__(self) -> str:
        if self.status == ExitStatus.SUCCESS:
            return f"feasible, cost = {self.cost()}"
        else:
            return "infeasible"


def main() -> None:
    # From section IV of [1]:

    # Initial lander mass including fuel (kg)
    m_wet = 2000.0

    # Initial fuel mass (kg)
    m_fuel = 300.0

    # Lander dry mass (kg)
    m_dry = m_wet - m_fuel

    # Maximum thrust (N)
    T_max = 24000

    ρ_1 = 0.2 * T_max
    ρ_2 = 0.8 * T_max

    # Fuel consumption rate (s/m)
    α = 5e-4
    assert α > 0

    # Initial position (m)
    q_0 = np.array([[2400.0, 450.0, -330.0]]).T

    # Initial velocity (m/s)
    v_0 = np.array([[-10.0, -40.0, 10.0]]).T

    # Final position (m)
    q_f = np.array([[0.0, 0.0, 0.0]]).T

    # Final velocity (m/s)
    v_f = np.array([[0.0, 0.0, 0.0]]).T

    # Gravitational acceleration on Mars (m/s²)
    g = np.array([[-3.71, 0.0, 0.0]]).T

    # Constant angular velocity of planet (rad/s)
    ω = np.array([[2.53e-5, 0.0, 6.62e-5]]).T

    # Thrust pointing limit (rad)
    θ = math.radians(90)
    assert θ >= 0 and θ <= math.pi

    # Minimum glide slope
    γ_gs = math.radians(30)
    assert γ_gs >= 0 and γ_gs <= math.pi / 2

    # Maximum velocity magnitude (m/s)
    v_max = 90.0

    # Time between control intervals (s)
    dt = 0.5

    # If ending straight, use different thrust magnitude constraints at end
    END_STRAIGHT = True

    # See equation (2) of [1].

    #     [ 0  -ω₃  ω₂]
    # S = [ ω₃  0  −ω₁]
    #     [−ω₂  ω₁  0 ]
    ω_1 = ω[0, 0]
    ω_2 = ω[1, 0]
    ω_3 = ω[2, 0]
    S = np.array([[0.0, -ω_3, ω_2], [ω_3, 0.0, -ω_1], [-ω_2, ω_1, 0.0]])

    #     [  0        I  ]
    # A = [-S(ω)²  -2S(ω)]
    A = np.block([[np.zeros((3, 3)), np.identity(3)], [-S @ S, -2 * S]])

    #     [0]
    # B = [I]
    B = np.block([[np.zeros((3, 3))], [np.identity(3)]])

    A_d, B_d, *_ = cont2discrete((A, B, np.identity(3), np.zeros((3, 3))), dt)

    def solve(N: int) -> Solution:
        problem = Problem()

        # x = [position, velocity]ᵀ
        X = problem.decision_variable(6, N + 1)
        # z = ln(m)
        Z = problem.decision_variable(1, N + 1)
        # u = T_c/m
        U = problem.decision_variable(3, N)
        # σ = Γ/m
        σ = problem.decision_variable(1, N)

        q = X[:3, :]
        v = X[3:6, :]

        # Initial position
        problem.subject_to(q[:, :1] == q_0)

        # Initial velocity
        problem.subject_to(v[:, :1] == v_0)

        # Initial mass
        problem.subject_to(Z[0, 0] == math.log(m_wet))

        # Final position
        problem.subject_to(q[:, -1] == q_f)

        # Final velocity
        problem.subject_to(v[:, -1] == v_f)

        # Final mass must be at least dry mass
        problem.subject_to(Z[0, -1] >= math.log(m_dry))

        # Position and velocity initial guesses
        for k in range(N + 1):
            for i in range(3):
                q[i, k].set_value(lerp(q_0[i, 0], q_f[i, 0], k / N))
                v[i, k].set_value(lerp(v_0[i, 0], v_f[i, 0], k / N))

        # State, input, and dynamics constraints
        for k in range(N + 1):
            t = k * dt

            x_k = X[:, k : k + 1]
            q_k = X[:3, k : k + 1]
            v_k = X[3:6, k : k + 1]
            z_k = Z[:, k : k + 1]

            # Velocity limits
            problem.subject_to(v_k.T @ v_k <= v_max**2)

            # Mass initial guess
            z_min = math.log(m_wet - α * ρ_2 * t)
            z_max = math.log(m_wet - α * ρ_1 * t)
            z_estimate = (z_min + z_max) / 2
            z_k.set_value(z_estimate)

            if k < N:
                x_k1 = X[:, k + 1 : k + 2]
                z_k1 = Z[:, k + 1 : k + 2]
                u_k = U[:, k : k + 1]
                σ_k = σ[:, k : k + 1]

                # Input initial guess
                #
                #   ρ₁ ≤ |T_c| ≤ ρ₂
                #   ρ₁ ≤ |u| exp(z) ≤ ρ₂
                #   ρ₁/exp(z) ≤ |u| ≤ ρ₂/exp(z)
                u_min = ρ_1 / math.exp(z_estimate)
                u_max = ρ_2 / math.exp(z_estimate)
                u_k.set_value(np.array([[(u_min + u_max) / 2, 0.0, 0.0]]).T)

                # Glide slope constraint on all but the final sample, which ensures
                # the trajectory isn't too shallow or goes below the target height
                #
                # See equation (12) of [1].
                #
                #       [0  1  0]
                #   E = [0  0  1]
                #
                #                      [1/tan(γ_gs)]
                #   c = e₁/tan(γ_gs) = [     0     ]
                #                      [     0     ]
                #
                #   |E(r - r_f)|₂ - cᵀ(r - r_f) ≤ 0                            (12)
                #
                #   hypot((r − r_f)₂, (r − r_f)₃) − (r − r_f)₁/tan(γ_gs) ≤ 0
                #   hypot((r − r_f)₂, (r − r_f)₃) ≤ (r − r_f)₁/tan(γ_gs)
                #   (r − r_f)₁/tan(γ_gs) ≥ hypot((r − r_f)₂, (r − r_f)₃)
                #   (r − r_f)₁²/tan²(γ_gs) ≥ (r − r_f)₂² + (r − r_f)₃²
                #   (r − r_f)₁² ≥ tan²(γ_gs)((r − r_f)₂² + (r − r_f)₃²)
                problem.subject_to(
                    (q_k[0, 0] - q_f[0, 0]) ** 2
                    >= math.tan(γ_gs) ** 2
                    * ((q_k[1, 0] - q_f[1, 0]) ** 2 + (q_k[2, 0] - q_f[2, 0]) ** 2)
                )

                problem.subject_to(σ_k >= 0)

                if k == N - 1 and END_STRAIGHT:
                    # Union of the following constraints:
                    #
                    #   uₖ[0] ≥ σₖ   x pointing constraint
                    #   uₖ[1] = 0    y pointing constraint
                    #   uₖ[2] = 0    z pointing constraint
                    #   uₖᵀuₖ ≤ σₖ²  thrust magnitude limit
                    problem.subject_to(u_k[0, 0] == σ_k)
                    problem.subject_to(u_k[1, 0] == 0)
                    problem.subject_to(u_k[2, 0] == 0)
                else:
                    # Thrust magnitude limit
                    #
                    # See equation (34) of [1].
                    #
                    #   |u|₂ ≤ σ
                    #   u_x² + u_y² + u_z² ≤ σ²
                    problem.subject_to(u_k.T @ u_k <= σ_k**2)

                    # Thrust pointing limit
                    #
                    # See equation (34) of [1].
                    #
                    #   n̂ᵀu ≥ cos(θ)σ where n̂ = [1  0  0]ᵀ
                    #   [1  0  0]u ≥ cos(θ)σ
                    #   u_x ≥ cos(θ)σ
                    problem.subject_to(u_k[0, 0] >= math.cos(θ) * σ_k)

                # Thrust slack limits
                #
                # See equation (34) of [2].
                z_0 = math.log(m_wet - α * ρ_2 * t)
                μ_1 = ρ_1 * math.exp(-z_0)
                μ_2 = ρ_2 * math.exp(-z_0)
                σ_min = μ_1 * (1 - (z_k - z_0) + 0.5 * (z_k - z_0) ** 2)
                σ_max = μ_2 * (1 - (z_k - z_0))
                problem.subject_to(bounds(σ_min, σ_k, σ_max))
                σ_k.set_value((σ_min.value() + σ_max.value()) / 2)

                # Integrate dynamics
                #
                # See equations (1), (2), and (33) of [1].
                #
                #   ẋ = Ax + B(g + u)
                #   ż = −ασ
                #
                #   xₖ₊₁ = A_d xₖ + B_d(g + uₖ)
                #   zₖ₊₁ = zₖ - αTσₖ
                problem.subject_to(x_k1 == A_d @ x_k + B_d @ (g + u_k))
                problem.subject_to(z_k1 == z_k - α * dt * σ_k)

        # Problem 4 from [1]: Minimum fuel
        problem.minimize(sum(σ))
        status = problem.solve()

        return Solution(status, X.value(), Z.value(), U.value(), σ.value())

    # Time horizon upper bound (s)
    #
    # The fuel caps the thrust's total velocity change at Δv_max.
    Δv_max = math.log(m_wet / m_dry) / α
    t_max = time_horizon_upper_bound(q_0, v_0, q_f, v_f, g, ω, Δv_max)

    # Time horizon lower bound (s)
    #
    # The thrust acceleration is at most ρ₂/m_dry.
    t_min = time_horizon_lower_bound(q_0, v_0, q_f, g, ω, v_max, ρ_2 / m_dry, θ, t_max)

    # Number of control intervals
    #
    # See equation (57) of [2].
    N_min = math.ceil(t_min / dt)
    N_max = math.floor(t_max / dt)

    # Find N with minimum fuel use
    print(f"Searching N ∈ [{N_min}, {N_max}] for minimum fuel use")
    N, sol = line_search(solve, N_min, N_max)
    print(f"N = {N}: {sol}")

    X_value = sol.X_value
    Z_value = sol.Z_value
    U_value = sol.U_value
    σ_value = sol.σ_value

    times = [k * dt for k in range(N + 1)]

    plt.figure()
    ax = plt.gca()
    ax.set_title("Position vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.plot(times, X_value[0, :], label="x (up)")
    ax.plot(times, X_value[1, :], label="y (east)")
    ax.plot(times, X_value[2, :], label="z (north)")
    ax.legend()

    plt.figure()
    ax = plt.gca()
    ax.set_title("Velocity vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.plot(times, X_value[3, :], label="x (up)")
    ax.plot(times, X_value[4, :], label="y (east)")
    ax.plot(times, X_value[5, :], label="z (north)")
    ax.plot(times, [norm(X_value[3:6, k]) for k in range(N + 1)], label="Total")
    ax.legend()

    plt.figure()
    ax = plt.gca()
    ax.set_title("Mass vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mass (kg)")
    ax.plot(times, [m_wet - α * ρ_2 * k * dt for k in range(N + 1)], label="Min mass")
    ax.plot(times, [m_wet - α * ρ_1 * k * dt for k in range(N + 1)], label="Max mass")
    ax.plot(times, [math.exp(Z_value[0, k]) for k in range(N + 1)], label="Mass")
    ax.legend()

    plt.figure()
    ax = plt.gca()
    ax.set_title("u vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("u")
    ax.plot(times[:-1], U_value[0, :], label="x (up)")
    ax.plot(times[:-1], U_value[1, :], label="y (east)")
    ax.plot(times[:-1], U_value[2, :], label="z (north)")
    ax.legend()

    plt.figure()
    ax = plt.gca()
    ax.set_title("σ vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("σ")
    ax.plot(times[:-1], σ_value.T)

    # Solve for |T_c|/T_max
    #
    # u = T_c/m
    # T_c = um
    # T_c = u exp(z)
    # |T_c| = |u| exp(z)
    # |T_c|/T_max = |u| exp(z)/T_max
    plt.figure()
    ax = plt.gca()
    ax.set_title("% Throttle vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("% Throttle")
    ax.plot(times[:-1], [ρ_1 / T_max * 100 for _ in range(N)], label="Min thrust")
    ax.plot(times[:-1], [ρ_2 / T_max * 100 for _ in range(N)], label="Max thrust")
    ax.plot(
        times[:-1],
        [norm(U_value[:, k]) * math.exp(Z_value[0, k]) / T_max * 100 for k in range(N)],
        label="Thrust",
    )
    ax.legend()

    plt.figure()
    ax = plt.gca()
    ax.set_title("Angle vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.plot(
        times,
        [
            math.degrees(
                math.atan2(X_value[0, k], math.hypot(X_value[1, k], X_value[2, k]))
            )
            for k in range(N + 1)
        ],
        label="Glide angle",
    )
    ax.plot(
        times[:-1],
        [
            math.degrees(
                math.acos(U_value[0, k] / norm(U_value[:, k]))
                if norm(U_value[:, k]) > 1e-2
                else 0
            )
            for k in range(N)
        ],
        label="Thrust angle from vertical",
    )
    ax.legend()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(vertical_axis="x")

    # Minimum glide slope cone
    h = q_0[0, 0] - q_f[0, 0]
    θs, Rs = np.meshgrid(
        np.linspace(0, 2 * math.pi, 100), np.linspace(0, h / math.tan(γ_gs), 100)
    )
    ax.plot_surface(
        Rs * math.tan(γ_gs) + q_f[2, 0],  # x
        Rs * np.sin(θs) + q_f[1, 0],  # y
        Rs * np.cos(θs) + q_f[0, 0],  # z
        alpha=0.2,
        label="Minimum glide slope",
    )

    # Rocket trajectory
    ax.plot(X_value[0, :], X_value[1, :], X_value[2, :], label="Trajectory")

    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
