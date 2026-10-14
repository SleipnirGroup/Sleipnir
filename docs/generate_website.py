#!/usr/bin/env python

import shutil
import subprocess
import sys
from pathlib import Path


def clear_python_workspace():
    subprocess.run(
        [
            "git",
            "restore",
            "../python/src/sleipnir/autodiff/__init__.py",
            "../python/src/sleipnir/optimization/__init__.py",
        ],
        check=True,
    )


def prep_python_api_docs():
    # Generate .pyi files
    subprocess.run(
        [
            "cmake",
            "-B",
            "build",
            "-S",
            "..",
            "-DCMAKE_C_COMPILER_LAUNCHER=sccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache",
            "-DSLEIPNIR_BUILD_PYTHON=ON",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", "build", "--target", "_sleipnir"], check=True)
    subprocess.run(
        [
            "cmake",
            "--install",
            "build",
            "--component",
            "python_modules",
            "--prefix",
            "build/install",
        ],
        check=True,
    )

    for package in ["autodiff", "optimization"]:
        # Read .pyi
        package_content = Path(
            "build", "install", "sleipnir", package, "__init__.pyi"
        ).read_text()

        # Remove redundant prefixes for documentation
        if package == "autodiff":
            package_content = package_content.replace(
                "import _sleipnir.optimization\n", ""
            ).replace("_sleipnir.optimization.", "")
        elif package == "optimization":
            package_content = package_content.replace(
                "import _sleipnir.autodiff\n", ""
            ).replace("_sleipnir.autodiff.", "")

        # Replace _sleipnir.package import with contents
        init_path = Path("../python/src/sleipnir", package, "__init__.py")
        init_path.write_text(
            init_path.read_text().replace(
                f"from .._sleipnir.{package} import *", package_content
            )
        )


def main():
    shutil.rmtree("build", ignore_errors=True)
    Path("build/html/docs").mkdir(parents=True, exist_ok=True)

    if "--doxygen-only" not in sys.argv:
        clear_python_workspace()
        prep_python_api_docs()

    subprocess.run(["doxygen", "Doxyfile-cpp"], check=True)
    subprocess.run(["doxygen", "Doxyfile-py"], check=True)
    subprocess.run(["doxygen", "Doxyfile"], check=True)

    if "--doxygen-only" not in sys.argv:
        clear_python_workspace()


if __name__ == "__main__":
    main()
