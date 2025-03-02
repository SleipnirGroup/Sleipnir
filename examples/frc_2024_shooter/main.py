#!/usr/bin/env python3

"""
FRC 2024 shooter trajectory optimization.

This program finds the initial velocity, pitch, and yaw for a game piece to hit
the 2024 FRC game's target that minimizes z sensitivity to initial velocity.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

from jormungandr.autodiff import Gradient, VariableMatrix
from jormungandr.optimization import Problem

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
g = 9.806  # m/s²


def f_variable(x):
    # x' = x'
    # y' = y'
    # z' = z'
    # x" = −a_D(v_x)
    # y" = −a_D(v_y)
    # z" = −g − a_D(v_z)
    #
    # where a_D(v) = ½ρv² C_D A / m
    # (see https://en.wikipedia.org/wiki/Drag_(physics)#The_drag_equation)
    rho = 1.204  # kg/m³
    C_D = 0.5
    A = math.pi * 0.3
    m = 2.0  # kg
    a_D = lambda v: 0.5 * rho * v**2 * C_D * A / m

    v_x = x[3, 0]
    v_y = x[4, 0]
    v_z = x[5, 0]
    return VariableMatrix(
        [[v_x], [v_y], [v_z], [-a_D(v_x)], [-a_D(v_y)], [-g - a_D(v_z)]]
    )


def f_double(x):
    # x' = x'
    # y' = y'
    # z' = z'
    # x" = −a_D(v_x)
    # y" = −a_D(v_y)
    # z" = −g − a_D(v_z)
    #
    # where a_D(v) = ½ρv² C_D A / m
    rho = 1.204  # kg/m³
    C_D = 0.5
    A = math.pi * 0.3
    m = 2.0  # kg
    a_D = lambda v: 0.5 * rho * v**2 * C_D * A / m

    v_x = x[3, 0]
    v_y = x[4, 0]
    v_z = x[5, 0]
    return np.array([[v_x], [v_y], [v_z], [-a_D(v_x)], [-a_D(v_y)], [-g - a_D(v_z)]])


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

    # Shooter initial position
    problem.subject_to(x[:3, :] == shooter_wrt_field[:3, :])

    # Require initial velocity is below max
    #
    #   √{v_x² + v_y² + v_z²) ≤ vₘₐₓ
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

    problem.solve(diagnostics=True)

    # Initial velocity vector
    v0 = x[3:, :].value() - robot_wrt_field[3:, :]

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
