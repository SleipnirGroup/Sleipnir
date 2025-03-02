#!/usr/bin/env python3

import math

import numpy as np

from jormungandr.optimization import Problem


def main():
    T = 5.0  # s
    dt = 0.005  # s
    N = int(T / dt)

    # Flywheel model:
    # States: [velocity]
    # Inputs: [voltage]
    A = math.exp(-dt)
    B = 1.0 - math.exp(-dt)

    problem = Problem()
    X = problem.decision_variable(1, N + 1)
    U = problem.decision_variable(1, N)

    # Dynamics constraint
    for k in range(N):
        problem.subject_to(
            X[:, k + 1 : k + 2] == A * X[:, k : k + 1] + B * U[:, k : k + 1]
        )

    # State and input constraints
    problem.subject_to(X[0, 0] == 0.0)
    problem.subject_to(-12 <= U)
    problem.subject_to(U <= 12)

    # Cost function - minimize error
    r = np.array([[10.0]])
    J = 0.0
    for k in range(N + 1):
        J += (r - X[:, k : k + 1]).T * (r - X[:, k : k + 1])
    problem.minimize(J)

    problem.solve()

    # The first state
    print(f"x₀ = {X.value(0, 0)}")

    # The first input
    print(f"u₀ = {U.value(0, 0)}")


if __name__ == "__main__":
    main()
