#!/usr/bin/env python3
"""Finds latest versions of the solution CSVs for each test problem, then plots
the time domain and X-Y data.

If provided, the first argument to this script is a filename regex that
restricts which CSVs are plotted to those that match the regex.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import re


class UnitGroup:
    def __init__(self):
        # List of DataSeries objects (time and data column pairs)
        self.series = []

        # List of data labels
        self.labels = []


class NameGroup:
    def __init__(self, filename, series):
        self.filename = filename
        self.series = series


class DataSeries:
    def __init__(self, time, data):
        self.time = time
        self.data = data


def num_lines(filename):
    with open(filename) as f:
        i = 0
        for i, l in enumerate(f):
            pass
    return i + 1


def get_file_list(regex):
    # Get list of files in current directory
    files = [
        os.path.join(dp, f)
        for dp, dn, fn in os.walk(".")
        for f in fn
        if f.endswith(".csv")
    ]

    # Ignore files not matching optional pattern
    if regex:
        files = [f for f in files if re.search(regex, f)]

    # Maps problem name to csv_group (group is set of files with states,
    # inputs, or outputs suffixes and the same name stub like "Flywheel")
    file_rgx = re.compile(r"(?P<name>^\./.*?[A-Za-z ]+)\.csv$")
    files = [f for f in files if file_rgx.search(f) and num_lines(f) > 2]

    return files


def make_groups(files):
    # Group files by category (sets of files with states, inputs, or outputs
    # suffixes and the same name stub like "Flywheel")
    category_rgx = re.compile(
        r"^\./.*?(?P<category>[A-Za-z\- ]+) (states|inputs|outputs)"
    )
    file_groups = {}
    if files:
        print("Loading CSVs...")
    else:
        print("No data to plot.")

    # Sorting the file list puts files into the order ["inputs", "outputs",
    # "states"]. This means data series will be loaded in the order of
    # ["inputs", "outputs", "references", "states"] (references are logged
    # before states). This produces the desired dataset layering on plots.
    for f in sorted(files):
        print(f"  {os.path.split(f)[1]}")

        match = category_rgx.search(f)

        if not match:
            # Couldn't find the file's category, so put it in its own category
            substr = os.path.split(f)[1]
            substr = substr[: substr.find("-")]
            file_groups[substr] = [f]
        else:
            category = match.group("category")

            # Create a new category if one doesn't exist, or add the file to the
            # existing category if it does
            if category not in file_groups.keys():
                file_groups[category] = [f]
            else:
                file_groups[category].append(f)
    return file_groups


def main():
    plt.rcParams.update({"figure.max_open_warning": 0})

    parser = argparse.ArgumentParser()
    parser.add_argument("-ymin", dest="ymin", type=float, help="Y minimum for plots")
    parser.add_argument("-ymax", dest="ymax", type=float, help="Y maximum for plots")
    parser.add_argument("regex", nargs="?")
    args = parser.parse_args()

    file_groups = make_groups(get_file_list(args.regex))
    if file_groups:
        print("Plotting...")

    # Within each group, make groups of datasets keyed on their unit, then plot
    # each group on their own figure
    unit_rgx = re.compile(r"^(?P<name>[\w\- ]+) \((?P<unit>.*?)\)$")
    for category, file_group in file_groups.items():
        unit_groups = {}
        name_groups = {}
        for filename in file_group:
            # Get labels from first row of file
            with open(filename) as f:
                labels = [x.strip('"') for x in f.readline().rstrip().split(",")]

            # Retrieve data from remaining rows of file. "skip_footer=1" skips
            # the last line because it may be incompletely written.
            data = np.genfromtxt(filename, delimiter=",", skip_header=1, skip_footer=1)

            times = data[:, 0:1]

            # Skips label in first column because that's always "Time (s)"
            for i, label in enumerate(labels[1:]):
                match = unit_rgx.search(label)
                name = match.group("name")
                unit = match.group("unit")

                if unit not in unit_groups.keys():
                    unit_groups[unit] = UnitGroup()
                # "i + 1" skips the time data column
                unit_groups[unit].series.append(
                    DataSeries(times, data[:, i + 1 : i + 2])
                )
                unit_groups[unit].labels.append(name)

                # "i + 1" skips the time data column
                name_groups[name] = NameGroup(filename, data[:, i + 1 : i + 2])

        # Plot time domain datasets
        print(f'  [vs time] {category} ({", ".join(unit_groups.keys())})')
        for unit, unit_group in unit_groups.items():
            fig, ax = plt.subplots(1, 1)
            ax.set_title(f"{category} ({unit})")

            for i in range(len(unit_group.series)):
                ax.plot(unit_group.series[i].time, unit_group.series[i].data)
                if args.ymin:
                    ax.set_ylim(bottom=args.ymin)
                if args.ymax:
                    ax.set_ylim(top=args.ymax)

            # First label is x axis label (time). The remainder are dataset
            # names.
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(f"Data ({unit})")
            ax.legend(unit_group.labels)

        # Plot X-Y datasets. If the file doesn't have all the required keys,
        # skip it.
        if not (
            set(["X reference", "Y reference", "X estimate", "Y estimate"])
            - set(name_groups.keys())
        ):
            print(f"  [y vs x] {category}")
            fig, ax = plt.subplots(1, 1)
            ax.set_title(f"{category} trajectory")

            ax.plot(
                name_groups["X reference"].series, name_groups["Y reference"].series
            )
            ax.plot(name_groups["X estimate"].series, name_groups["Y estimate"].series)

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.legend(["Reference", "Estimate"])

            # This equalizes the X and Y axes so the trajectories aren't warped
            ax.axis("equal")

    plt.show()


if __name__ == "__main__":
    main()
