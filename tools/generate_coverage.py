#!/usr/bin/env python

"""
Generates merged coverage of CMake target executables with llvm-cov.
"""

import argparse
import os
import subprocess
from pathlib import Path

BUILD_DIR = Path("build-coverage")
IGNORE_REGEX = "_deps/|test/"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="CMake target executables to run",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="name used for merged coverage output files",
    )
    args = parser.parse_args()

    # Build executables
    subprocess.run(
        [
            "cmake",
            "-B",
            BUILD_DIR,
            "-S",
            ".",
            "-DCMAKE_BUILD_TYPE=Coverage",
            "-DCMAKE_C_COMPILER=clang",
            "-DCMAKE_CXX_COMPILER=clang++",
            "-DCMAKE_C_COMPILER_LAUNCHER=sccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache",
        ],
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            BUILD_DIR,
            "--target",
            *args.inputs,
            "--parallel",
            str(os.cpu_count()),
        ],
        check=True,
    )

    # Run executables, each writing its own raw profile
    profraws = []
    for target in args.inputs:
        profraw = f"{target}.profraw"
        subprocess.run(
            [f"./{target}"],
            cwd=BUILD_DIR,
            env=os.environ | {"LLVM_PROFILE_FILE": profraw},
            stdout=subprocess.DEVNULL,
            check=True,
        )
        profraws.append(profraw)

    # Merge raw profiles
    profdata = f"{args.output}.profdata"
    subprocess.run(
        ["llvm-profdata", "merge", "-sparse", *profraws, "-o", profdata],
        cwd=BUILD_DIR,
        check=True,
    )

    # Prepare arguments for llvm-cov. llvm-cov takes the first executable
    # positionally and the rest via -object.
    objects = [f"./{args.inputs[0]}"]
    for target in args.inputs[1:]:
        objects += ["-object", f"./{target}"]
    cov_args = [
        f"-ignore-filename-regex={IGNORE_REGEX}",
        *objects,
        f"-instr-profile={profdata}",
    ]

    # Generate reports
    with open(BUILD_DIR / f"coverage-line-by-line-{args.output}.html", "w") as f:
        subprocess.run(
            ["llvm-cov", "show", *cov_args, "-format=html"],
            cwd=BUILD_DIR,
            stdout=f,
            check=True,
        )
    with open(BUILD_DIR / f"coverage-report-{args.output}.txt", "w") as f:
        subprocess.run(
            ["llvm-cov", "report", *cov_args], cwd=BUILD_DIR, stdout=f, check=True
        )


if __name__ == "__main__":
    main()
