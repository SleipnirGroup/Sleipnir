#!/usr/bin/env python3

import os
import platform
import shutil
import subprocess
import sys


def main():
    # Clear workspace
    shutil.rmtree(".py-build-cmake_cache", ignore_errors=True)
    subprocess.run(
        [
            "git",
            "restore",
            "jormungandr/autodiff/__init__.py",
            "jormungandr/optimization/__init__.py",
        ],
        check=True,
    )

    # Generate .pyi files
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation"], check=True
    )

    version_tuple = platform.python_version_tuple()
    version_num = f"{version_tuple[0]}{version_tuple[1]}"
    PYI_PATH = os.path.join(
        ".py-build-cmake_cache",
        # e.g., "cp312-cp312-linux_x86_64-stubs"
        f"cp{version_num}-cp{version_num}-{platform.system().lower()}_{platform.machine()}-stubs",
        "_jormungandr",
    )

    for package in ["autodiff", "optimization"]:
        # Read .pyi
        with open(os.path.join(PYI_PATH, package + ".pyi")) as f:
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


if __name__ == "__main__":
    main()
