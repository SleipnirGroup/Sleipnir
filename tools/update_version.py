#!/usr/bin/env python3

"""Updates version string in pyproject.toml."""

import re
import subprocess


def main():
    output = subprocess.check_output(["git", "describe"], encoding="utf-8").rstrip()
    print(f"git describe: {output}")

    m = re.search(r"^v(\d+)\.(\d+)\.(\d+)(-(\d+))?", output)
    major = m.group(1)
    minor = m.group(2)
    patch = m.group(3)
    commits_since_tag = m.group(5)
    if commits_since_tag:
        version = f"{major}.{minor}.{int(patch) + 1}.dev{commits_since_tag}"
    else:
        version = f"{major}.{minor}.{patch}"
    print(f'__version__ = "{version}"')

    with open("python/src/sleipnir/__init__.py") as f:
        content = f.read()
    content = re.sub(r"(__version__\s*=\s*)\".*?\"", f'\\g<1>"{version}"', content)
    with open("python/src/sleipnir/__init__.py", "w") as f:
        f.write(content)


if __name__ == "__main__":
    main()
