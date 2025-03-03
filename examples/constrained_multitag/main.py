#!/usr/bin/env python3

"""
Determines a robot pose from the corner pixel locations of several AprilTags.

The robot pose is constrained to be on the floor (z = 0).
"""

import numpy as np

from jormungandr.autodiff import Variable, VariableMatrix, cos, sin, solve
from jormungandr.optimization import Problem


def main():
    problem = Problem()

    # camera calibration
    fx = 600
    fy = 600
    cx = 300
    cy = 150

    # robot pose
    robot_x = problem.decision_variable()
    robot_y = problem.decision_variable()
    robot_z = Variable(0)
    robot_θ = problem.decision_variable()

    # cache autodiff variables
    sinθ = sin(robot_θ)
    cosθ = cos(robot_θ)

    var0 = Variable(0)
    var1 = Variable(1)
    field2robot = VariableMatrix(
        [
            [cosθ, -sinθ, var0, robot_x],
            [sinθ, cosθ, var0, robot_y],
            [var0, var0, var1, robot_z],
            [var0, var0, var0, var1],
        ]
    )

    # robot is ENU, cameras are SDE
    robot2camera = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

    field2camera = field2robot @ robot2camera

    # list of points in field space to reproject. Each one is a 4x1 vector of
    # (x,y,z,1)
    field2points = [
        VariableMatrix([[2, 0 - 0.08255, 0.4, 1]]).T,
        VariableMatrix([[2, 0 + 0.08255, 0.4, 1]]).T,
    ]

    # List of points we saw the target at. These are exactly what we expect for
    # a camera located at 0,0,0 (hand-calculated)
    point_observations = [(325, 30), (275, 30)]

    # initial guess at robot pose. We expect the robot to converge to 0,0,0
    robot_x.set_value(-0.1)
    robot_y.set_value(0.0)
    robot_θ.set_value(0.2)

    # field2camera * field2camera⁻¹ = I
    camera2field = solve(field2camera, VariableMatrix(np.identity(4)))

    # Cost
    J = 0
    for field2point, observation in zip(field2points, point_observations):
        # camera2point = field2camera⁻¹ * field2point
        # field2camera * camera2point = field2point
        camera2point = camera2field @ field2point

        # point's coordinates in camera frame
        x = camera2point[0]
        y = camera2point[1]
        z = camera2point[2]

        print(f"camera2point = {x.value()}, {y.value()}, {z.value()}")

        # coordinates observed at
        u_observed, v_observed = observation

        X = x / z
        Y = y / z

        u = fx * X + cx
        v = fy * Y + cy

        print(f"Expected u {u.value()}, saw {u_observed}")
        print(f"Expected v {v.value()}, saw {v_observed}")

        u_err = u - u_observed
        v_err = v - v_observed

        # Cost function is square of reprojection error
        J += u_err**2 + v_err**2

    problem.minimize(J)

    problem.solve(diagnostics=True)

    print(f"x = {robot_x.value()} m")
    print(f"y = {robot_y.value()} m")
    print(f"θ = {robot_θ.value()} rad")


if __name__ == "__main__":
    main()
