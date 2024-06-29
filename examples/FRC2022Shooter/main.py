#!/usr/bin/env python3

"""
FRC 2022 shooter trajectory optimization.

This program finds the optimal initial launch velocity and launch angle for the
2022 FRC game's target.
"""

import math

from jormungandr.autodiff import VariableMatrix
from jormungandr.optimization import OptimizationProblem
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import norm

field_width = 8.2296  # 27 ft
field_length = 16.4592  # 54 ft
g = 9.806  # m/s²


def main():
    # Robot initial velocity
    robot_initial_v_x = 0.2  # ft/s
    robot_initial_v_y = -0.2  # ft/s
    robot_initial_v_z = 0.0  # ft/s

    max_launch_velocity = 10.0

    shooter = np.array([[field_length / 4.0], [field_width / 4.0], [1.2]])
    shooter_x = shooter[0, 0]
    shooter_y = shooter[1, 0]
    shooter_z = shooter[2, 0]

    target = np.array([[field_length / 2.0], [field_width / 2.0], [2.64]])
    target_x = target[0, 0]
    target_y = target[1, 0]
    target_z = target[2, 0]
    target_radius = 0.61

    def lerp(a, b, t):
        return a + t * (b - a)

    problem = OptimizationProblem()

    # Set up duration decision variables
    N = 10
    T = problem.decision_variable()
    problem.subject_to(T >= 0)
    T.set_value(1)
    dt = T / N

    #     [x position]
    #     [y position]
    #     [z position]
    # x = [x velocity]
    #     [y velocity]
    #     [z velocity]
    X = problem.decision_variable(6, N)

    p_x = X[0, :]
    p_y = X[1, :]
    p_z = X[2, :]
    v_x = X[3, :]
    v_y = X[4, :]
    v_z = X[5, :]

    # Position initial guess is linear interpolation between start and end position
    for k in range(N):
        p_x[k].set_value(lerp(shooter_x, target_x, k / N))
        p_y[k].set_value(lerp(shooter_y, target_y, k / N))
        p_z[k].set_value(lerp(shooter_z, target_z, k / N))

    # Velocity initial guess is max launch velocity toward goal
    uvec_shooter_to_target = target - shooter
    uvec_shooter_to_target /= norm(uvec_shooter_to_target)
    for k in range(N):
        v_x[k].set_value(max_launch_velocity * uvec_shooter_to_target[0, 0])
        v_y[k].set_value(max_launch_velocity * uvec_shooter_to_target[1, 0])
        v_z[k].set_value(max_launch_velocity * uvec_shooter_to_target[2, 0])

    # Shooter initial position
    problem.subject_to(X[:3, 0] == shooter)

    # Require initial launch velocity is below max
    #
    #   √{v_x² + v_y² + v_z²) ≤ vₘₐₓ
    #   v_x² + v_y² + v_z² ≤ vₘₐₓ²
    problem.subject_to(
        v_x[0] ** 2 + v_y[0] ** 2 + v_z[0] ** 2 <= max_launch_velocity**2
    )

    # Require final position is in center of target circle
    problem.subject_to(p_x[-1] == target_x)
    problem.subject_to(p_y[-1] == target_y)
    problem.subject_to(p_z[-1] == target_z)

    # Require the final velocity is down
    problem.subject_to(v_z[-1] < 0.0)

    def f(x):
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

        v_x = x[3, 0] + robot_initial_v_x
        v_y = x[4, 0] + robot_initial_v_y
        v_z = x[5, 0] + robot_initial_v_z
        return VariableMatrix(
            [[v_x], [v_y], [v_z], [-a_D(v_x)], [-a_D(v_y)], [-g - a_D(v_z)]]
        )

    # Dynamics constraints - RK4 integration
    for k in range(N - 1):
        h = dt
        x_k = X[:, k]
        x_k1 = X[:, k + 1]

        k1 = f(x_k)
        k2 = f(x_k + h / 2 * k1)
        k3 = f(x_k + h / 2 * k2)
        k4 = f(x_k + h * k3)
        problem.subject_to(x_k1 == x_k + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))

    # Minimize time to goal
    problem.minimize(T)

    problem.solve(diagnostics=True)

    # Initial velocity vector
    v = X[3:, 0].value()

    launch_velocity = norm(v)
    print(f"Launch velocity = {launch_velocity:.03f} m/s")

    pitch = math.atan2(v[2, 0], math.hypot(v[0, 0], v[1, 0]))
    print(f"Pitch = {pitch * 180.0 / math.pi:.03f}°")

    yaw = math.atan2(v[1, 0], v[0, 0])
    print(f"Yaw = {yaw * 180.0 / math.pi:.03f}°")

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
        target_x,
        target_y,
        target_z,
        color="black",
        marker="x",
    )
    xs = []
    ys = []
    zs = []
    for angle in np.arange(0.0, 2.0 * math.pi, 0.1):
        xs.append(target_x + target_radius * math.cos(angle))
        ys.append(target_y + target_radius * math.sin(angle))
        zs.append(target_z)
    ax.plot(xs, ys, zs, color="black")

    # Trajectory
    trajectory_x = p_x.value()[0, :]
    trajectory_y = p_y.value()[0, :]
    trajectory_z = p_z.value()[0, :]
    ax.plot(trajectory_x, trajectory_y, trajectory_z, color="orange")

    ax.set_box_aspect((field_length, field_width, np.max(trajectory_z)))

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_zlabel("Z position (m)")

    plt.show()


if __name__ == "__main__":
    main()
