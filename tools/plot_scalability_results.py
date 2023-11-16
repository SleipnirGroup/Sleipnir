#!/usr/bin/env python3

import argparse
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


def main():
    parser = argparse.ArgumentParser(
        description="Runs all formatting tasks on the code base. This should be invoked from a directory within the project."
    )
    parser.add_argument(
        "--filenames",
        dest="filenames",
        type=str,
        nargs="+",
        required=True,
        help="filenames of CSVs to plot",
    )
    parser.add_argument(
        "--labels",
        dest="labels",
        type=str,
        nargs="+",
        required=True,
        help="plot labels for each filename's data",
    )
    parser.add_argument(
        "--title",
        dest="title",
        type=str,
        required=True,
        help="plot title",
    )
    parser.add_argument(
        "--noninteractive",
        dest="noninteractive",
        action="store_true",
        help="if present, saves the figure to a file instead of displaying it",
    )
    args = parser.parse_args()

    samples = []
    setup_times = []
    solve_times = []

    for filename in args.filenames:
        (
            sample,
            setup_time,
            solve_time,
        ) = np.genfromtxt(
            filename,
            delimiter=",",
            skip_header=1,
            unpack=True,
            invalid_raise=False,
            ndmin=2,
        )
        if math.isnan(setup_time[-1]) or math.isnan(solve_time[-1]):
            sample = sample[:-1]
            setup_time = setup_time[:-1]
            solve_time = solve_time[:-1]
        if len(sample) == 0:
            print("No data to plot.")
            sys.exit(1)

        samples.append(sample)
        setup_times.append(setup_time)
        solve_times.append(solve_time)

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_title(f"{args.title} problem runtime vs samples")
    ax1.set_ylabel("Setup time (ms)")
    ax1.grid(visible=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i in range(len(samples)):
        ax1.plot(samples[i], setup_times[i], label=args.labels[i])
        plot_poly2_fit(ax1, samples[i], setup_times[i], color=colors[i])

    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_xlabel("Samples")
    ax2.set_ylabel("Solve time (ms)")
    ax2.grid(visible=True)

    for i in range(len(samples)):
        ax2.plot(samples[i], solve_times[i], label=args.labels[i])
        plot_exp2_fit(
            ax2, samples[i], solve_times[i], color=colors[i], force_intercept=True
        )

    ax2.legend()

    if args.noninteractive:
        plt.savefig(f"{args.title.lower()}-scalability-results.png")
    else:
        plt.show()


if __name__ == "__main__":
    main()
