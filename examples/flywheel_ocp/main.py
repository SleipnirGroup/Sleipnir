#!/usr/bin/env python3

import math

import numpy as np
from sleipnir.autodiff import VariableMatrix
from sleipnir.optimization import (
    OCP,
    DynamicsType,
    TimestepMethod,
    TranscriptionMethod,
)


def main():
    T = 5.0  # s
    dt = 0.005  # s
    N = int(T / dt)

    # Flywheel model:
    # States: [velocity]
    # Inputs: [voltage]
    A = -1.0
    B = 1.0

    A_discrete = math.exp(A * dt)
    B_discrete = (1.0 - A_discrete) * B

    def f_discrete(x: VariableMatrix, u: VariableMatrix):
        return A_discrete * x + B_discrete * u

    r = 10.0

    solver = OCP(
        1,
        1,
        dt,
        N,
        f_discrete,
        DynamicsType.DISCRETE,
        TimestepMethod.FIXED,
        TranscriptionMethod.DIRECT_TRANSCRIPTION,
    )
    solver.constrain_initial_state(0.0)
    solver.set_upper_input_bound(12)
    solver.set_lower_input_bound(-12)

    # Set up cost
    r_mat = np.full((1, N + 1), r)
    solver.minimize((r_mat - solver.X()) @ (r_mat - solver.X()).T)

    solver.solve()

    # The first state
    print(f"x₀ = {solver.X().value(0, 0)}")

    # The first input
    print(f"u₀ = {solver.U().value(0, 0)}")


if __name__ == "__main__":
    main()
