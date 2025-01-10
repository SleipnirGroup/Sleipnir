#!/usr/bin/env python3

import argparse
import struct
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import animation
from matplotlib.colors import ListedColormap


def plot_csv(filename: str) -> animation.FuncAnimation:
    print(f"Loading sparsity pattern {filename}...", end="")
    sys.stdout.flush()

    xs = []
    ys = []
    vs = []

    x = []
    y = []
    v = []

    with open(filename, mode="rb") as f:
        size = struct.unpack("<i", f.read(4))[0]
        title = struct.unpack(f"{size}s", f.read(size))[0].decode("utf-8")

        size = struct.unpack("<i", f.read(4))[0]
        row_label = struct.unpack(f"{size}s", f.read(size))[0].decode("utf-8")

        size = struct.unpack("<i", f.read(4))[0]
        col_label = struct.unpack(f"{size}s", f.read(size))[0].decode("utf-8")

        rows = struct.unpack("<i", f.read(4))[0]
        cols = struct.unpack("<i", f.read(4))[0]

        try:
            while True:
                num_coords = struct.unpack("<i", f.read(4))[0]
                for _ in range(num_coords):
                    y.append(struct.unpack("<i", f.read(4))[0])
                    x.append(struct.unpack("<i", f.read(4))[0])
                    sign = struct.unpack("c", f.read(1))[0]
                    if sign == b"+":
                        v.append(1.0)
                    elif sign == b"-":
                        v.append(-1.0)
                    else:
                        v.append(0.0)
                xs.append(x)
                ys.append(y)
                vs.append(v)
                x = []
                y = []
                v = []
        except:
            pass

    print(" done.")
    sys.stdout.flush()

    fig = plt.figure()
    ax = fig.add_subplot()

    # Display scatter plot
    cmap = ListedColormap(["red", "black", "green"])
    sc = ax.scatter(xs[0], ys[0], s=1, c=vs[0], marker=".", cmap=cmap, vmin=-1, vmax=1)
    iteration = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    # Display colorbar
    colorbar = fig.colorbar(sc)
    ticklabels = ["Negative", "Positive"]
    colorbar.ax.yaxis.set_major_locator(mticker.FixedLocator(ticklabels))
    colorbar.ax.set_yticks([-1, 1])
    colorbar.ax.set_yticklabels(ticklabels)

    ax.set_title(title)
    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)

    ax.set_xlim([0, cols])
    ax.set_ylim([0, rows])
    ax.invert_yaxis()
    ax.set_aspect(1.0)

    def animate(i: int):
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
    parser = argparse.ArgumentParser(
        description="Displays sparsity pattern (.spy) files."
    )
    parser.add_argument("filename", nargs="+")
    args = parser.parse_args()

    # pragma pylint: disable=unused-variable
    animations = [plot_csv(filename) for filename in args.filename]  # noqa
    plt.show()


if __name__ == "__main__":
    main()
