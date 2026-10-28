#!/usr/bin/env python

"""
Plots autodiff and optimization benchmark results from CSV files.
"""

import argparse
import csv
import sys
from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.optimize import nnls

# Conversion factors from Google Benchmark time units to milliseconds
MS_PER_UNIT = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}

# (median, minimum, maximum) times
type Stats = tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]


def load_results(
    filename: str, prefix: str, kinds: list[str]
) -> tuple[npt.NDArray[np.int64], *tuple[Stats, ...]]:
    """
    Loads times from a Google Benchmark CSV file for the benchmarks named
    "<prefix>/<kind>/<N>" for each kind.

    If the benchmarks were repeated, the median, minimum, and maximum of the
    repetitions are used. Otherwise, the single run is used for all three.
    Benchmarks that reported an error are ignored.

    Args:
        filename: The CSV filename.
        prefix: The benchmark name prefix (e.g., "Sleipnir").
        kinds: The kinds of time to load (e.g., ["Setup", "Evaluation"]).

    Returns:
        Tuple of N values followed by (median times (ms), minimum times (ms),
        maximum times (ms)) for each kind.
    """
    with open(filename) as f:
        # Skip the context lines that precede the header
        lines = iter(f)
        for line in lines:
            if line.startswith("name,"):
                break
        benchmarks = list(csv.DictReader(lines, fieldnames=line.strip().split(",")))

    # Ignore benchmarks that reported an error
    benchmarks = [b for b in benchmarks if b["error_occurred"] != "true"]

    # Prefer aggregates over individual runs if they exist. Aggregates have the
    # aggregate name appended to the run name (e.g., "_median").
    stats = ("_median", "_min", "_max")
    aggregates = ("_mean", "_stddev", "_cv", *stats)
    if any(b["name"].endswith("_median") for b in benchmarks):
        benchmarks = [b for b in benchmarks if b["name"].endswith(stats)]
    else:
        benchmarks = [b for b in benchmarks if not b["name"].endswith(aggregates)]

    # Names have the form "<prefix>/<kind>/<N>[/<options>][_<aggregate>]",
    # where the options are "manual_time" and "<key>:<value>" pairs for
    # explicitly set benchmark options (e.g., "repeats:3")
    times: dict[str, dict[str, dict[int, float]]] = {
        stat: {kind: {} for kind in kinds} for stat in stats
    }
    for benchmark in benchmarks:
        name = benchmark["name"]

        # Aggregate suffix (e.g., "_median"), or None for a single run
        stat = next((stat for stat in stats if name.endswith(stat)), None)

        # Strip the aggregate suffix and options, leaving the prefix components,
        # kind, and N
        parts = [
            part
            for part in name.removesuffix(stat or "").split("/")
            if part != "manual_time" and ":" not in part
        ]

        # The prefix may itself contain slashes, so take the last two parts as
        # the kind and N and rejoin the rest
        *name_prefix, kind, N = parts
        if "/".join(name_prefix) == prefix and kind in kinds:
            # Normalize the time to milliseconds
            time = float(benchmark["real_time"]) * MS_PER_UNIT[benchmark["time_unit"]]

            # An aggregate fills only its own stat; the other stats come from
            # sibling aggregate rows. A single run is its own median, minimum,
            # and maximum.
            for s in [stat] if stat else stats:
                times[s][kind][int(N)] = time

    # Keep only the N values that have every stat for every kind so the arrays
    # line up
    Ns = sorted(
        set.intersection(
            *(set(times[stat][kind].keys()) for stat in stats for kind in kinds)
        )
    )

    # Converts a kind's times to (median, minimum, maximum) arrays ordered by N
    def to_stats(kind: str) -> Stats:
        median, minimum, maximum = (
            np.array([times[stat][kind][N] for N in Ns], dtype=np.float64)
            for stat in stats
        )
        return median, minimum, maximum

    return np.array(Ns, dtype=np.int64), *(to_stats(kind) for kind in kinds)


def plot_poly_fit(
    ax: Axes,
    x: npt.NDArray[np.int64],
    y: npt.NDArray[np.float64],
    basis_funcs: list[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
    basis_labels: list[str],
    color: str,
) -> None:
    """
    Plots a least-squares fit of a linear combination of basis functions.

    Args:
        ax: The axis on which to plot.
        x: The list of x values.
        y: The list of y values.
        basis_funcs: The basis functions of x to combine. Each must be
            nondecreasing for x ≥ 1.
        basis_labels: A list of strings to print for the basis terms.
        color: The color to use for the plot.
    """

    # Evaluates the basis functions as the columns of a matrix
    def basis_matrix(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.column_stack([f(x) for f in basis_funcs])

    # Fit the coefficients with nonnegative least squares since runtime can't
    # decrease with problem size. Scale the columns to unit maximum so the solve
    # is well-conditioned.
    A = basis_matrix(x.astype(np.float64))
    scale = np.max(np.abs(A), axis=0)
    coeffs = nnls(A / scale, y)[0] / scale

    # Build the fit equation label, omitting terms whose largest contribution
    # over the data range is below 0.1% of the data's magnitude. Such terms are
    # visually indistinguishable from zero and far below measurement noise. The
    # coefficients are nonnegative and the bases are nondecreasing, so each
    # term's contribution peaks at the largest x.
    contributions = coeffs * A[-1]
    label = " + ".join(
        f"{coeff:.4g}{basis_label}"
        for coeff, contribution, basis_label in zip(coeffs, contributions, basis_labels)
        if contribution > 1e-3 * np.max(np.abs(y))
    )
    label = "Fit: y = " + (label or "0")

    # Evaluate the fit on a finer grid spanning the data so the curve is smooth
    resampled_x = np.arange(x[0], x[-1] + 10, 10, dtype=np.float64)
    ax.plot(
        resampled_x,
        basis_matrix(resampled_x) @ coeffs,
        color=color,
        label=label,
        linestyle="--",
    )


def plot_results(
    results: list[tuple[npt.NDArray[np.int64], Sequence[Stats]]],
    labels: list[str],
    title: str,
    xlabel: str,
    ylabels: list[str],
) -> Figure:
    """
    Plots benchmark results with one subplot per kind of time.

    Each subplot has a quadratic curve fit for each data series.

    Args:
        results: A list of (x, ys) tuples for each data series, where ys is a
            tuple of (y median array, y minimum array, y maximum array) tuples
            with one per subplot. The minimum-maximum ranges are drawn as error
            bars.
        labels: The plot label for each data series.
        title: The figure title.
        xlabel: The x-axis label.
        ylabels: The y-axis label for each subplot.

    Returns:
        The figure.
    """
    # Use the default color cycle so each curve fit matches its data series
    fig = plt.figure()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for row, ylabel in enumerate(ylabels):
        # Stack the subplots vertically with the title on the top one and the
        # x-axis label on the bottom one
        ax = fig.add_subplot(len(ylabels), 1, row + 1)
        if row == 0:
            ax.set_title(title)
        if row == len(ylabels) - 1:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(visible=True)

        # Plot each data series with min-max error bars and a quadratic fit of
        # its medians
        for i, (x, ys) in enumerate(results):
            y, y_min, y_max = ys[row]
            ax.errorbar(x, y, yerr=(y - y_min, y_max - y), label=labels[i], capsize=3)
            plot_poly_fit(
                ax,
                x,
                y,
                [lambda x: x**2, lambda x: x, lambda x: np.ones_like(x)],
                ["x²", "x", ""],
                color=colors[i],
            )

        # Matplotlib puts error bar entries after line entries, so reorder the
        # legend to list each data series before its curve fit
        handles, legend_labels = ax.get_legend_handles_labels()
        n = len(results)
        order = [j for i in range(n) for j in (n + i, i)]
        ax.legend([handles[j] for j in order], [legend_labels[j] for j in order])

    return fig


def main() -> None:
    # Options shared by all benchmark types
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--noninteractive",
        dest="noninteractive",
        action="store_true",
        help="if present, saves the figure to a file instead of displaying it",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="benchmark", required=True)

    autodiff_parser = subparsers.add_parser(
        "autodiff", parents=[common_parser], help="plot autodiff benchmark results"
    )
    autodiff_parser.add_argument(
        "--filename",
        dest="filename",
        type=str,
        required=True,
        help="filename of Google Benchmark CSV results to plot",
    )
    autodiff_parser.add_argument(
        "--libraries",
        dest="libraries",
        type=str,
        nargs="+",
        default=["CasADi", "Sleipnir"],
        help="autodiff libraries whose results to plot",
    )
    autodiff_parser.add_argument(
        "--title",
        dest="title",
        type=str,
        required=True,
        help="plot title",
    )

    optimization_parser = subparsers.add_parser(
        "optimization",
        parents=[common_parser],
        help="plot optimization benchmark results",
    )
    optimization_parser.add_argument(
        "--filename",
        dest="filename",
        type=str,
        required=True,
        help="filename of Google Benchmark CSV results to plot",
    )
    optimization_parser.add_argument(
        "--libraries",
        dest="libraries",
        type=str,
        nargs="+",
        default=["CasADi", "Sleipnir"],
        help="optimization libraries whose results to plot",
    )
    optimization_parser.add_argument(
        "--labels",
        dest="labels",
        type=str,
        nargs="+",
        help="plot labels for each library's data (defaults to library names)",
    )
    optimization_parser.add_argument(
        "--title",
        dest="title",
        type=str,
        required=True,
        help="plot title",
    )

    args = parser.parse_args()

    if args.benchmark == "autodiff":
        kinds = ["Setup", "Evaluation"]
        prefixes = args.libraries
        labels = args.libraries
        title = f"{args.title} autodiff runtime vs variables"
        xlabel = "Variables"
        output = f"{args.title.lower()}-autodiff-results.png"
    elif args.benchmark == "optimization":
        kinds = ["Setup", "Solve"]
        prefixes = args.libraries
        if args.labels is None:
            default_labels = {"CasADi": "CasADi + Ipopt + MUMPS"}
            labels = [default_labels.get(library, library) for library in prefixes]
        else:
            labels = args.labels
        title = f"{args.title} problem runtime vs samples"
        xlabel = "Samples"
        output = f"{args.title.lower()}-optimization-results.png"

    if len(labels) != len(prefixes):
        print(f"Got {len(prefixes)} libraries but {len(labels)} labels.")
        sys.exit(1)

    # Load each library's results, failing if any library has none
    results: list[tuple[npt.NDArray[np.int64], Sequence[Stats]]] = []
    for prefix in prefixes:
        x, *times = load_results(args.filename, prefix, kinds)
        if len(x) == 0:
            print(f"No {prefix} data to plot in {args.filename}.")
            sys.exit(1)
        results.append((x, times))

    fig = plot_results(
        results, labels, title, xlabel, [f"{kind} time (ms)" for kind in kinds]
    )

    if args.noninteractive:
        fig.savefig(output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
