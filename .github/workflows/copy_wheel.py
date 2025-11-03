#!/usr/bin/env python3

import argparse
import re
import shutil


def main():
    parser = argparse.ArgumentParser(
        description="Duplicates a stable ABI wheel for the given Python versions."
    )
    parser.add_argument("wheel", help="Wheel to duplicate")
    parser.add_argument("versions", nargs="+", help="List of Python versions")
    args = parser.parse_args()

    for version in args.versions:
        dest = re.sub(r"cp[0-9]+", f"cp{version.replace(".", "")}", args.wheel)
        shutil.copyfile(args.wheel, dest)


if __name__ == "__main__":
    main()
