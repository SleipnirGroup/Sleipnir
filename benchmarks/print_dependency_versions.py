#!/usr/bin/env python

"""
Prints the CasADi, Ipopt, linear solver, and BLAS versions used by the
optimization benchmarks.

The benchmark binary is run once with the dynamic linker's library tracing
enabled, so the versions are those of the libraries it actually loads at
runtime. This requires glibc (LD_DEBUG) on Linux.
"""

import argparse
import ctypes
import os
import re
import subprocess
import sys
from pathlib import Path


def run_benchmark(binary: Path) -> tuple[str, list[Path]]:
    """
    Runs one CasADi solve with diagnostics enabled.

    Returns the Ipopt banner line and the list of loaded shared libraries.
    """
    result = subprocess.run(
        [binary, "--enable-diagnostics"],
        env=dict(os.environ, LD_DEBUG="libs"),
        capture_output=True,
        text=True,
        check=True,
    )

    banner = re.search(r"This is Ipopt version .*", result.stdout)
    if not banner:
        sys.exit("error: Ipopt banner not found in benchmark output")

    libs = sorted(
        {Path(lib) for lib in re.findall(r"calling init: (\S+)", result.stderr)}
    )
    return banner.group(0), libs


def find_lib(libs: list[Path], *names: str) -> Path | None:
    """Returns the first loaded library whose filename starts with a name."""
    for name in names:
        for lib in libs:
            if lib.name.startswith(name):
                return lib
    return None


def casadi_version(libs: list[Path]) -> str:
    lib = find_lib(libs, "libcasadi.so")
    if lib is None:
        return "unknown"

    # Demangles to casadi::CasadiMeta::version()
    f = ctypes.CDLL(lib)._ZN6casadi10CasadiMeta7versionEv
    f.restype = ctypes.c_char_p
    return f().decode()


def blas_version(libs: list[Path]) -> str:
    lib = find_lib(libs, "libopenblas")
    if lib is not None:
        f = ctypes.CDLL(lib).openblas_get_config
        f.restype = ctypes.c_char_p
        # e.g., "OpenBLAS 0.3.34 DYNAMIC_ARCH ..." -> "OpenBLAS 0.3.34"
        return " ".join(f().decode().split()[:2])

    # Fall back to the resolved filename of an unrecognized BLAS
    lib = find_lib(libs, "libblas", "libblis", "libcblas")
    if lib is not None:
        return lib.resolve().name
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "build/flywheel_optimization_benchmark",
        help="optimization benchmark binary to inspect",
    )
    args = parser.parse_args()

    banner, libs = run_benchmark(args.binary)

    # Example output:
    # "This is Ipopt version 3.14.20, running with linear solver MUMPS 5.8.1."
    m = re.match(
        r"This is Ipopt version ([^,]+), running with linear solver (.+)\.$", banner
    )
    if not m:
        sys.exit(f"error: couldn't parse Ipopt banner: {banner}")
    ipopt, linear_solver = m.groups()

    print(f"* CasADi {casadi_version(libs)} (autodiff and NLP solver frontend)")
    print(f"* Ipopt {ipopt} (NLP solver backend)")
    print(f"* {linear_solver} (linear solver)")
    print(f"* {blas_version(libs)} (BLAS)")


if __name__ == "__main__":
    main()
