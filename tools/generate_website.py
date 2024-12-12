#!/usr/bin/env python3

import os
import shutil
import subprocess
import sys


def clear_python_workspace():
    shutil.rmtree("build-stubs", ignore_errors=True)
    subprocess.run(
        [
            "git",
            "restore",
            "jormungandr/autodiff/__init__.py",
            "jormungandr/optimization/__init__.py",
        ],
        check=True,
    )


def prep_python_api_docs():
    # Generate .pyi files
    subprocess.run(
        ["cmake", "-B", "build-stubs", "-S", ".", "-DBUILD_PYTHON=ON"], check=True
    )
    subprocess.run(
        ["cmake", "--build", "build-stubs", "--target", "_jormungandr"], check=True
    )
    subprocess.run(
        [
            "cmake",
            "--install",
            "build-stubs",
            "--component",
            "python_modules",
            "--prefix",
            "build-stubs/install",
        ],
        check=True,
    )

    for package in ["autodiff", "optimization"]:
        # Read .pyi
        with open(
            os.path.join(
                "build-stubs", "install", "jormungandr", package, "__init__.pyi"
            )
        ) as f:
            package_content = f.read()

        # Remove redundant prefixes for documentation
        if package == "autodiff":
            package_content = package_content.replace(
                "import _jormungandr.optimization\n", ""
            ).replace("_jormungandr.optimization.", "")
        elif package == "optimization":
            package_content = package_content.replace(
                "import _jormungandr.autodiff\n", ""
            ).replace("_jormungandr.autodiff.", "")

        # Replace _jormungandr.package import with contents
        with open(os.path.join("jormungandr", package, "__init__.py")) as f:
            init_content = f.read()
        with open(os.path.join("jormungandr", package, "__init__.py"), mode="w") as f:
            f.write(
                init_content.replace(
                    f"from .._jormungandr.{package} import *", package_content
                )
            )


def main():
    shutil.rmtree("build/html", ignore_errors=True)
    os.makedirs("build/html/docs", exist_ok=True)

    if "--doxygen-only" not in sys.argv:
        clear_python_workspace()
        prep_python_api_docs()

    subprocess.run(
        [
            "doxygen",
            "docs/Doxyfile-cpp",
        ],
        check=True,
    )
    subprocess.run(
        [
            "doxygen",
            "docs/Doxyfile-py",
        ],
        check=True,
    )
    subprocess.run(
        [
            "doxygen",
            "docs/Doxyfile",
        ],
        check=True,
    )

    if "--doxygen-only" not in sys.argv:
        clear_python_workspace()


if __name__ == "__main__":
    main()
