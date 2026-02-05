#!/usr/bin/env python3

"""
FRC 2024 shooter trajectory optimization.

This program finds the initial velocity, pitch, and yaw for a game piece to hit
the 2024 FRC game's target that minimizes either z sensitivity to initial
velocity or initial velocity (see minimize() calls below).

This optimization problem formulation uses single-shooting on the flight
dynamics, including air resistance, to allow minimizing z sensitivity.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from sleipnir.autodiff import Gradient, VariableMatrix, block, sqrt
from sleipnir.optimization import Problem

field_width = 8.2296  # 27 ft -> m
field_length = 16.4592  # 54 ft -> m
target_width = 1.05  # m
target_lower_edge = 1.98  # m
target_upper_edge = 2.11  # m
target_depth = 0.46  # m
target_wrt_field = np.array(
    [
        [field_length - target_depth / 2],
        [field_width - 2.6575],
        [(target_upper_edge + target_lower_edge) / 2],
        [0.0],
        [0.0],
        [0.0],
    ]
)
g = np.array([[0], [0], [9.806]])  # m/s²


def cross_variable(a, b) -> VariableMatrix:
    return VariableMatrix(
        [
            [a[1, 0] * b[2, 0] - a[2, 0] * b[1, 0]],
            [a[2, 0] * b[0, 0] - a[0, 0] * b[2, 0]],
            [a[0, 0] * b[1, 0] - a[1, 0] * b[0, 0]],
        ]
    )


def f_variable(x):
    # x' = x'
    # y' = y'
    # z' = z'
    # [x"]   [ 0]
    # [y"] = [ 0] − F_D(v)/m v̂ − F_L(v)/m (ω x v)
    # [z"]   [−g]

    # ρ is the fluid density in kg/m³
    # v is the linear velocity in m/s
    # v̂ is the velocity direction unit vector
    # ω is the angular velocity in rad/s
    # A is the cross-sectional area of a circle in m²
    # m is the object mass in kg
    ρ = 1.204  # kg/m³
    v = x[3:6, :]  # m/s
    v2 = (v.T @ v)[0, 0]
    v_norm = sqrt(v2)
    v_hat = v / v_norm
    ω = np.array([[0.0], [0.0], [2.0]])  # rad/s
    r = 0.15  # m
    A = math.pi * r**2  # m²
    m = 0.283  # kg

    # Per https://en.wikipedia.org/wiki/Drag_(physics)#The_drag_equation:
    #   F_D(v) = ½ρ|v|²C_D A
    #   C_D is the drag coefficient (dimensionless)
    C_D = 0.5
    F_D = 0.5 * ρ * v2 * C_D * A

    # Magnus force:
    #   F_L(v) = ½ρ|v|C_L A
    #   C_L is the lift coefficient (dimensionless)
    C_L = 0.5
    F_L = 0.5 * ρ * v_norm * C_L * A

    return block([[v], [-g - F_D / m * v_hat - F_L / m * cross_variable(v, ω)]])


def cross_double(a, b):
    return np.array(
        [
            [a[1, 0] * b[2, 0] - a[2, 0] * b[1, 0]],
            [a[2, 0] * b[0, 0] - a[0, 0] * b[2, 0]],
            [a[0, 0] * b[1, 0] - a[1, 0] * b[0, 0]],
        ]
    )


def f_double(x):
    # x' = x'
    # y' = y'
    # z' = z'
    # [x"]   [ 0]
    # [y"] = [ 0] − F_D(v)/m v̂ − F_L(v)/m (ω x v)
    # [z"]   [−g]

    # ρ is the fluid density in kg/m³
    # v is the linear velocity in m/s
    # v̂ is the velocity direction unit vector
    # ω is the angular velocity in rad/s
    # A is the cross-sectional area of a circle in m²
    # m is the object mass in kg
    ρ = 1.204  # kg/m³
    v = x[3:6, :]  # m/s
    v2 = (v.T @ v)[0, 0]
    v_norm = math.sqrt(v2)
    v_hat = v / v_norm
    ω = np.array([[0.0], [0.0], [2.0]])  # rad/s
    r = 0.15  # m
    A = math.pi * r**2  # m²
    m = 0.283  # kg

    # Per https://en.wikipedia.org/wiki/Drag_(physics)#The_drag_equation:
    #   F_D(v) = ½ρ|v|²C_D A
    #   C_D is the drag coefficient (dimensionless)
    C_D = 0.5
    F_D = 0.5 * ρ * v2 * C_D * A

    # Magnus force:
    #   F_L(v) = ½ρ|v|C_L A
    #   C_L is the lift coefficient (dimensionless)
    C_L = 0.5
    F_L = 0.5 * ρ * v_norm * C_L * A

    return np.block([[v], [-g - F_D / m * v_hat - F_L / m * cross_double(v, ω)]])


def main():
    # Robot initial state
    robot_wrt_field = np.array(
        [[0.75 * field_length], [field_width / 3.0], [0.0], [1.524], [-1.524], [0.0]]
    )

    max_initial_velocity = 15.0  # m/s

    shooter_wrt_robot = np.array([[0.0], [0.0], [0.6096], [0.0], [0.0], [0.0]])
    shooter_wrt_field = robot_wrt_field + shooter_wrt_robot

    problem = Problem()

    # Set up duration decision variables
    N = 10
    T = problem.decision_variable()
    problem.subject_to(T >= 0)
    T.set_value(1)
    dt = T / N

    # Disc state in field frame
    #
    #     [x position]
    #     [y position]
    #     [z position]
    # x = [x velocity]
    #     [y velocity]
    #     [z velocity]
    x = problem.decision_variable(6)

    # Position initial guess is start position
    x[:3, :].set_value(shooter_wrt_field[:3, :])

    # Velocity initial guess is max initial velocity toward target
    uvec_shooter_to_target = target_wrt_field[:3, :] - shooter_wrt_field[:3, :]
    uvec_shooter_to_target /= norm(uvec_shooter_to_target)
    x[3:, :].set_value(
        robot_wrt_field[3:, :] + max_initial_velocity * uvec_shooter_to_target
    )

    v0_wrt_shooter = x[3:, :] - shooter_wrt_field[3:, :]

    # Shooter initial position
    problem.subject_to(x[:3, :] == shooter_wrt_field[:3, :])

    # Require initial velocity is below max
    #
    #   √(v_x² + v_y² + v_z²) ≤ vₘₐₓ
    #   v_x² + v_y² + v_z² ≤ vₘₐₓ²
    problem.subject_to(
        (x[3] - robot_wrt_field[3, 0]) ** 2
        + (x[4] - robot_wrt_field[4, 0]) ** 2
        + (x[5] - robot_wrt_field[5, 0]) ** 2
        <= max_initial_velocity**2
    )

    # Single-shooting - RK4 integration
    h = dt
    x_k = x
    for _ in range(N - 1):
        k1 = f_variable(x_k)
        k2 = f_variable(x_k + h / 2 * k1)
        k3 = f_variable(x_k + h / 2 * k2)
        k4 = f_variable(x_k + h * k3)
        x_k += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # Require final position is in center of target circle
    problem.subject_to(x_k[:3, :] == target_wrt_field[:3, :])

    # Require the final velocity is up
    problem.subject_to(x_k[5, 0] > 0.0)

    # Minimize sensitivity of vertical position to velocity
    sensitivity = Gradient(x_k[3, 0], x[3:, :]).get()
    problem.minimize(sensitivity.T @ sensitivity)

    # Minimize initial velocity
    # problem.minimize(v0_wrt_shooter.T @ v0_wrt_shooter)

    problem.solve(diagnostics=True)

    # Initial velocity vector with respect to shooter
    v0 = v0_wrt_shooter.value()

    velocity = norm(v0)
    print(f"Velocity = {velocity:.03f} m/s")

    pitch = math.atan2(v0[2, 0], math.hypot(v0[0, 0], v0[1, 0]))
    print(f"Pitch = {np.rad2deg(pitch):.03f}°")

    yaw = math.atan2(v0[1, 0], v0[0, 0])
    print(f"Yaw = {np.rad2deg(yaw):.03f}°")

    print(f"Total time = {T.value():.03f} s")
    print(f"dt = {dt.value() * 1e3:.03f} ms")

    plt.figure()
    ax = plt.axes(projection="3d")

    def plot_wireframe(ax, f, x_range, y_range, color):
        x, y = np.mgrid[x_range[0] : x_range[1] : 25j, y_range[0] : y_range[1] : 25j]

        # Need an (N, 2) array of (x, y) pairs.
        xy = np.column_stack([x.flat, y.flat])

        z = np.zeros(xy.shape[0])
        for i, pair in enumerate(xy):
            z[i] = f(pair[0], pair[1])
        z = z.reshape(x.shape)

        ax.plot_wireframe(x, y, z, color=color)

    # Ground
    plot_wireframe(ax, lambda x, y: 0.0, [0, field_length], [0, field_width], "grey")

    # Target
    ax.plot(
        target_wrt_field[0, 0],
        target_wrt_field[1, 0],
        target_wrt_field[2, 0],
        color="black",
        marker="x",
    )
    xs = []
    ys = []
    zs = []
    # Bottom-left corner
    xs.append(field_length)
    ys.append(target_wrt_field[1, 0] - target_width / 2)
    zs.append(target_lower_edge)
    # Bottom-right corner
    xs.append(field_length)
    ys.append(target_wrt_field[1, 0] + target_width / 2)
    zs.append(target_lower_edge)
    # Top-right corner
    xs.append(field_length - target_depth)
    ys.append(target_wrt_field[1, 0] + target_width / 2)
    zs.append(target_upper_edge)
    # Top-left corner
    xs.append(field_length - target_depth)
    ys.append(target_wrt_field[1, 0] - target_width / 2)
    zs.append(target_upper_edge)
    # Bottom-left corner
    xs.append(field_length)
    ys.append(target_wrt_field[1, 0] - target_width / 2)
    zs.append(target_lower_edge)
    ax.plot(xs, ys, zs, color="black")

    # Trajectory
    trajectory_x = []
    trajectory_y = []
    trajectory_z = []
    h = dt.value()
    x_k = x.value()
    trajectory_x.append(x_k[0, 0])
    trajectory_y.append(x_k[1, 0])
    trajectory_z.append(x_k[2, 0])
    for _ in range(N - 1):
        k1 = f_double(x_k)
        k2 = f_double(x_k + h / 2 * k1)
        k3 = f_double(x_k + h / 2 * k2)
        k4 = f_double(x_k + h * k3)
        x_k += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        trajectory_x.append(x_k[0, 0])
        trajectory_y.append(x_k[1, 0])
        trajectory_z.append(x_k[2, 0])
    ax.plot(trajectory_x, trajectory_y, trajectory_z, color="orange")

    ax.set_box_aspect((field_length, field_width, np.max(trajectory_z)))

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_zlabel("Z position (m)")

    plt.show()


if __name__ == "__main__":
    main()
