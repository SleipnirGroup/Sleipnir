def rk4(f, x, u, dt):
    """
    Performs 4th order Runge-Kutta integration of dx/dt = f(x, u) for dt.

    Parameter ``f``:
        The function to integrate. It must take two arguments x and u.

    Parameter ``x``:
        The initial value of x.

    Parameter ``u``:
        The value u held constant over the integration period.

    Parameter ``dt``:
        The time over which to integrate.
    """
    h = dt

    k1 = f(x, u)
    k2 = f(x + h * 0.5 * k1, u)
    k3 = f(x + h * 0.5 * k2, u)
    k4 = f(x + h * k3, u)

    return x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
