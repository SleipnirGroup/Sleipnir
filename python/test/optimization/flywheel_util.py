import numpy as np
import numpy.typing as npt


def optimal_flywheel_trajectory(
    A: float, B: float, r: float, u_max: float, N: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Returns the states x₀, …, x_N and inputs u₀, …, u_N₋₁ that minimize
    Σ(r − xₖ)² for k = 0, …, N subject to xₖ₊₁ = Axₖ + Buₖ, x₀ = 0, and
    |uₖ| ≤ u_max, where 0 < A < 1, B > 0, r > 0, and u_max ≥ (1 − A)r/B
    (the input that holds xₖ = r).

    Parameter ``A``:
        Discrete state transition.

    Parameter ``B``:
        Discrete input gain.

    Parameter ``r``:
        Reference state.

    Parameter ``u_max``:
        Input bound.

    Parameter ``N``:
        Number of timesteps.
    """
    # Every xₖ increases with each of u₀, …, uₖ₋₁, so each cost term is
    # minimized by driving xₖ as close to r as the input bound allows.
    # Saturating the input until one step can land exactly on r, then holding it
    # there, minimizes every term at once.
    #
    # Once xₖ = r, the landing input is uₖ = (r − Ar)/B = (1 − A)r/B, the
    # steady-state input, so the landing branch also holds the reference.

    X = np.zeros(N + 1)
    U = np.zeros(N)

    for k in range(N):
        x = X[k]

        if A * x + B * u_max < r:
            # Saturate the input until the reference is within one step
            U[k] = u_max
            X[k + 1] = A * x + B * u_max
        else:
            # Land exactly on (or hold) the reference
            U[k] = (r - A * x) / B
            X[k + 1] = r

    return X, U
