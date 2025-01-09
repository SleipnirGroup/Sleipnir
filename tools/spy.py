#!/usr/bin/env python3

"""Loads and displays the sparsity patterns from A_e.spy, A_i.spy, and H.spy."""

import re
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import animation
from matplotlib.colors import ListedColormap


def plot_csv(filename, title, xlabel, ylabel):
    print(f"Loading sparsity pattern for {title}...", end="")
    sys.stdout.flush()

    with open(filename) as f:
        contents = f.read()

    max_row_idx = 0
    max_col_idx = 0
    row = 0
    col = 0

    xs = []
    ys = []
    vs = []

    x = []
    y = []
    v = []

    for m in re.finditer(r"\.|\+|\-|\n\n|\n", contents):
        token = m.group()

        if token == ".":
            # Do nothing since zero entries aren't logged
            col += 1
        elif token == "+":
            # Log positive entry
            x.append(col)
            y.append(row)
            v.append(1.0)
            col += 1
        elif token == "-":
            # Log negative entry
            x.append(col)
            y.append(row)
            v.append(-1.0)
            col += 1
        elif token == "\n\n":
            # Prep for new matrix. Log old one if it's not truncated.
            if row >= max_row_idx and col >= max_col_idx:
                max_row_idx = row
                max_col_idx = col

                xs.append(x)
                ys.append(y)
                vs.append(v)
            x = []
            y = []
            v = []
            row = 0
            col = 0
        elif token == "\n" and m.span()[0] < len(contents) - 1:
            # Prep for new row if it isn't the last
            row += 1
            col = 0

    # Log leftover matrix if it's not truncated
    if row >= max_row_idx and col >= max_col_idx:
        max_row_idx = row
        max_col_idx = col

        xs.append(x)
        ys.append(y)
        vs.append(v)

    print(" done.")
    sys.stdout.flush()

    fig = plt.figure()
    ax = fig.add_subplot()

    # Display scatter plot
    cmap = ListedColormap(["red", "green"])
    sc = ax.scatter(xs[0], ys[0], s=1, c=vs[0], marker=".", cmap=cmap, vmin=-1, vmax=1)
    iteration = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    # Display colorbar
    colorbar = fig.colorbar(sc)
    ticklabels = ["Negative", "Positive"]
    colorbar.ax.yaxis.set_major_locator(mticker.FixedLocator(ticklabels))
    colorbar.ax.set_yticks([-1, 1])
    colorbar.ax.set_yticklabels(ticklabels)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xlim([0, max_col_idx])
    ax.set_ylim([0, max_row_idx])
    ax.invert_yaxis()
    ax.set_aspect(1.0)

    def animate(i):
        sc.set_offsets(np.c_[xs[i], ys[i]])
        sc.set_array(vs[i])
        iteration.set_text(f"iter {i}/{max(0, len(vs) - 1)}")
        return (sc, iteration)

    return animation.FuncAnimation(
        fig=fig,
        func=animate,
        frames=len(vs),
        interval=1.0 / 30.0,
        repeat_delay=1000,
        repeat=True,
        blit=True,
    )


def main():
    # pragma pylint: disable=unused-variable
    anim1 = plot_csv(
        "A_e.spy",
        title="Equality constraint Jacobian",
        xlabel="Decision variables",
        ylabel="Constraints",
    )
    # pragma pylint: disable=unused-variable
    anim2 = plot_csv(
        "A_i.spy",
        title="Inequality constraint Jacobian",
        xlabel="Decision variables",
        ylabel="Constraints",
    )
    # pragma pylint: disable=unused-variable
    anim3 = plot_csv(
        "H.spy",
        title="Hessian",
        xlabel="Decision variables",
        ylabel="Decision variables",
    )
    plt.show()


if __name__ == "__main__":
    main()
