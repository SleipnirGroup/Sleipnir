"""Test tolerances that follow the error model of the value being checked.

See "Choosing test tolerances" in docs/contributing.md for how to choose the
error model and multiple. Python only tests double, so take them from the C++
version of the test, which also tests float.
"""

import math
import sys
from enum import Enum, auto

ε = sys.float_info.epsilon

# Default solver tolerance
TOLERANCE = math.sqrt(ε)


def rounding_tolerance(multiple: float) -> float:
    """
    Returns the tolerance for a value whose error is dominated by rounding error,
    a multiple of ε.

    Use this for values computed without the solver (e.g., autodiff
    derivatives). Values the solver produces are only as accurate as its
    termination tolerance, so use solution_tolerance() for those, even if their
    error is usually much smaller.

    Parameter ``multiple``:
        Error bound in multiples of machine epsilon.
    """
    return multiple * ε


class SolutionType(Enum):
    """
    How a solution's error depends on the solver's termination tolerance tol.

    The solver stops once the KKT error is below tol.
    """

    REGULAR = auto()
    """
    LICQ, strict complementarity, and second-order sufficiency hold, so the
    solution error is linear in the KKT error, O(tol).
    """

    DEGENERATE = auto()
    """
    One of those fails, so the solution error is O(√tol). For example, a constraint
    is weakly active (active with a zero Lagrange multiplier), there are no finite
    Lagrange multipliers, or the cost is nearly flat in a feasible direction (e.g.,
    an input mode that barely affects the cost).
    """


def solution_tolerance(
    multiple: float, type: SolutionType = SolutionType.REGULAR
) -> float:
    """
    Returns the tolerance for a value whose error is bounded by the solver's
    termination tolerance, a multiple of tol for a regular solution or √tol for a
    degenerate solution.

    tol is the default solver tolerance, √ε. For a regular solution, the multiple
    is the sensitivity of the value to the KKT error.

    Equality constraint residuals (e.g., initial and final states, and dynamics
    defects) are part of the KKT error, so the solver guarantees they're at most
    tol. Use a multiple of at least 1 for them; their error is often much smaller,
    but how much smaller depends on the iterates' path, which varies across
    platforms.

    Parameter ``multiple``:
        Error bound in multiples of tol or √tol.

    Parameter ``type``:
        How the solution error depends on tol.
    """
    if type == SolutionType.DEGENERATE:
        return multiple * math.sqrt(TOLERANCE)
    return multiple * TOLERANCE
