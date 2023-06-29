#!/usr/bin/env python3

"""Updates version string in pyproject.toml."""

import re
import subprocess


def main():
    proc = subprocess.run(
        [
            "git",
            "describe",
        ],
        encoding="utf-8",
        stdout=subprocess.PIPE,
    )
    if proc.returncode:
        version = "0.0.0"
    else:
        m = re.search(
            r"^v ([0-9]+\.[0-9]+\.[0-9]+) (- ([0-9]+) )?",
            proc.stdout.rstrip(),
            re.X,
        )

        # Version number: <tag>.dev<# commits since tag>
        version = m.group(1)
        if m.group(2):
            version += f".dev{m.group(3)}"

    # Update version string in pyproject.toml
    with open("pyproject.toml") as f:
        content = f.read()
    content = re.sub(r"\b(version\s*=\s*)\".*?\"", r"\g<1>" + f'"{version}"', content)
    with open("pyproject.toml", "w") as f:
        f.write(content)


if __name__ == "__main__":
    main()
