#!/usr/bin/env python3

import os
import platform
import shutil
import subprocess
import sys


def main():
    # Clear workspace
    shutil.rmtree(".py-build-cmake_cache", ignore_errors=True)

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
            pyi_content = f.read()

        # Fix up contents
        if package == "autodiff":
            pyi_content = pyi_content.replace("sleipnir::", "").replace(
                "VariableBlock<VariableMatrix>", "VariableBlock"
            )
        elif package == "optimization":
            pyi_content = (
                pyi_content.replace("_jormungandr.autodiff.", "")
                .replace("import _jormungandr.autodiff\n", "")
                .replace("<ExpressionType.NONE: 0>", "ExpressionType.NONE")
                .replace(
                    "<SolverExitCondition.SUCCESS: 0>", "SolverExitCondition.SUCCESS"
                )
            )

        # Replace _jormungandr.package import with contents
        with (
            open(os.path.join("jormungandr", package, "__init__.py")) as input,
            open(
                os.path.join("jormungandr", package, "__init__.py.new"), mode="w"
            ) as output,
        ):
            package_content = input.read()
            output.write(
                package_content.replace(
                    f"from .._jormungandr.{package} import *", pyi_content
                )
            )
        shutil.move(
            os.path.join("jormungandr", package, "__init__.py.new"),
            os.path.join("jormungandr", package, "__init__.py"),
        )


if __name__ == "__main__":
    main()
