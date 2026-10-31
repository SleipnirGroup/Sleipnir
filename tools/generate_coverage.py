#!/usr/bin/env python

"""
Generates merged coverage of CMake target executables with llvm-cov (Clang) or
gcovr (GCC).
"""

import argparse
import os
import subprocess
from pathlib import Path

BUILD_DIR = Path("build-coverage")
IGNORE_REGEX = "_deps/|test/"

COMPILERS = {"clang": ("clang", "clang++"), "gcc": ("gcc", "g++")}


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
    parser.add_argument(
        "--compiler",
        choices=COMPILERS.keys(),
        default="clang",
        help="compiler to build with; gcc requires gcovr",
    )
    args = parser.parse_args()
    cc, cxx = COMPILERS[args.compiler]

    # Build executables. Configure from scratch so switching compilers doesn't
    # discard the cache entries below.
    subprocess.run(
        [
            "cmake",
            "--fresh",
            "-B",
            BUILD_DIR,
            "-S",
            ".",
            "-DCMAKE_BUILD_TYPE=Coverage",
            f"-DCMAKE_C_COMPILER={cc}",
            f"-DCMAKE_CXX_COMPILER={cxx}",
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

    if args.compiler == "clang":
        generate_llvm_cov_reports(args.inputs, args.output)
    else:
        generate_gcovr_reports(args.inputs, args.output)


def generate_llvm_cov_reports(inputs: list[str], output: str) -> None:
    # Run executables, each writing its own raw profile
    profraws = []
    for target in inputs:
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
    profdata = f"{output}.profdata"
    subprocess.run(
        ["llvm-profdata", "merge", "-sparse", *profraws, "-o", profdata],
        cwd=BUILD_DIR,
        check=True,
    )

    # Prepare arguments for llvm-cov. llvm-cov takes the first executable
    # positionally and the rest via -object.
    objects = [f"./{inputs[0]}"]
    for target in inputs[1:]:
        objects += ["-object", f"./{target}"]
    cov_args = [
        f"-ignore-filename-regex={IGNORE_REGEX}",
        *objects,
        f"-instr-profile={profdata}",
    ]

    # Generate reports
    with open(BUILD_DIR / f"coverage-line-by-line-{output}.html", "w") as f:
        subprocess.run(
            ["llvm-cov", "show", *cov_args, "-format=html"],
            cwd=BUILD_DIR,
            stdout=f,
            check=True,
        )
    with open(BUILD_DIR / f"coverage-report-{output}.txt", "w") as f:
        subprocess.run(
            ["llvm-cov", "report", *cov_args], cwd=BUILD_DIR, stdout=f, check=True
        )


def generate_gcovr_reports(inputs: list[str], output: str) -> None:
    # gcov counters accumulate across runs, so remove those from previous runs
    for gcda in BUILD_DIR.rglob("*.gcda"):
        gcda.unlink()

    # Run executables, each adding to the shared gcov counters
    for target in inputs:
        subprocess.run(
            [f"./{target}"], cwd=BUILD_DIR, stdout=subprocess.DEVNULL, check=True
        )

    # Generate reports
    subprocess.run(
        [
            "gcovr",
            "--root",
            ".",
            f"--exclude=.*({IGNORE_REGEX})",
            # Catch2's -ffile-prefix-map makes its sources look like they're
            # in the build directory
            f"--exclude={BUILD_DIR}/",
            "--html-details",
            BUILD_DIR / f"coverage-line-by-line-{output}.html",
            "--html-single-page",
            "--txt",
            BUILD_DIR / f"coverage-report-{output}.txt",
            "-j",
            str(os.cpu_count()),
            BUILD_DIR,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
