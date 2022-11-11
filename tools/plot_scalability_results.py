#!/usr/bin/env python3

import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def plot_poly2_fit(ax, x, y, color):
    def poly2(x, a, b, c):
        return a * x**2 + b * x + c

    # Fit 2nd degree polynomial y = ax² + bx + c to x-y data
    a, b, c = curve_fit(poly2, x, y, p0=(1, 1, 1), bounds=([0, 0, 0], np.inf))[0]

    label = f"Fit: y = {a:.4g}x²"
    if b > 0:
        label += f" + {b:.4g}x"
    else:
        label += f" - {abs(b):.4g}x"
    if c > 0:
        label += f" + {c:.4g}"
    else:
        label += f" - {abs(c):.4g}"

    resampled_x = np.arange(x[0], x[-1] + 100, 100)
    ax.plot(
        resampled_x,
        poly2(resampled_x, a, b, c),
        color=color,
        label=label,
        linestyle="--",
    )


def plot_exp2_fit(ax, x, y, color, force_intercept=False):
    if not force_intercept:

        def exp2(x, a, b, c):
            return a * np.exp2(b * x) + c

        # Fit exponential y = a2ᵇˣ + c to x-y data
        a, b, c = curve_fit(exp2, x, y, p0=(1, 1e-6, 1))[0]

        label = f"Fit: y = {a:.4g} 2^({b:.4g}x)"
        if c > 0:
            label += f" + {c:.4g}"
        else:
            label += f" - {abs(c):.4g}"

        resampled_x = np.arange(x[0], x[-1] + 100, 100)
        ax.plot(
            resampled_x,
            exp2(resampled_x, a, b, c),
            color=color,
            label=label,
            linestyle="--",
        )
    else:

        def exp2(x, a, b):
            return a * (np.exp2(b * x) - 1)

        # Fit exponential y = c(2ᵇˣ − 1) to x-y data
        a, b = curve_fit(exp2, x, y, p0=(1, 1e-6))[0]

        resampled_x = np.arange(x[0], x[-1] + 100, 100)
        ax.plot(
            resampled_x,
            exp2(resampled_x, a, b),
            color=color,
            label=f"Fit: y = {a:.4g} (2^({b:.4g}x) - 1)",
            linestyle="--",
        )


(
    samples,
    casadi_setup_time,
    casadi_solve_time,
    problem_setup_time,
    problem_solve_time,
) = np.genfromtxt(
    "scalability-results.csv",
    delimiter=",",
    skip_header=1,
    unpack=True,
    invalid_raise=False,
    ndmin=2,
)

if (
    math.isnan(casadi_setup_time[-1])
    or math.isnan(casadi_solve_time[-1])
    or math.isnan(problem_setup_time[-1])
    or math.isnan(problem_solve_time[-1])
):
    samples = samples[:-1]
    casadi_setup_time = casadi_setup_time[:-1]
    casadi_solve_time = casadi_solve_time[:-1]
    problem_setup_time = problem_setup_time[:-1]
    problem_solve_time = problem_solve_time[:-1]

if len(samples) == 0:
    print("No data to plot.")
    sys.exit(1)

fig = plt.figure()

ax1 = fig.add_subplot(2, 1, 1)
ax1.set_title("Optimization API runtime vs samples")
ax1.set_ylabel("Setup time (ms)")
ax1.grid(visible=True)

ax1.plot(samples, casadi_setup_time, label="CasADi")
plot_poly2_fit(ax1, samples, casadi_setup_time, color="blue")

ax1.plot(samples, problem_setup_time, label="Sleipnir")
plot_poly2_fit(ax1, samples, problem_setup_time, color="orange")

ax1.legend()

ax2 = fig.add_subplot(2, 1, 2)
ax2.set_xlabel("Samples")
ax2.set_ylabel("Solve time (ms)")
ax2.grid(visible=True)

ax2.plot(samples, casadi_solve_time, label="CasADi")
plot_exp2_fit(ax2, samples, casadi_solve_time, color="blue", force_intercept=True)

ax2.plot(samples, problem_solve_time, label="Sleipnir")
plot_exp2_fit(ax2, samples, problem_solve_time, color="orange", force_intercept=True)

ax2.legend()

plt.show()
