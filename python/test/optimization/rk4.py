from collections.abc import Callable
from typing import Any, overload

import numpy as np
import numpy.typing as npt
from sleipnir.autodiff import VariableBlock, VariableMatrix

type Matrix = npt.NDArray[np.float64]
type SymbolicMatrix = VariableMatrix | VariableBlock


@overload
def rk4(
    f: Callable[[Matrix, Matrix], Matrix], x: Matrix, u: Matrix, dt: float
) -> Matrix: ...


@overload
def rk4(
    f: Callable[[SymbolicMatrix, SymbolicMatrix], VariableMatrix],
    x: SymbolicMatrix,
    u: SymbolicMatrix,
    dt: float,
) -> VariableMatrix: ...


def rk4(f: Callable[[Any, Any], Any], x: Any, u: Any, dt: float) -> Any:
    """
    Performs 4th order Runge-Kutta integration of dx/dt = f(x, u) for dt.

    Args:
        f: The function to integrate. It must take two arguments x and u.
        x: The initial value of x.
        u: The value u held constant over the integration period.
        dt: The time over which to integrate.
    """
    h = dt

    k1 = f(x, u)
    k2 = f(x + h * 0.5 * k1, u)
    k3 = f(x + h * 0.5 * k2, u)
    k4 = f(x + h * k3, u)

    return x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
