#!/usr/bin/env python

import argparse
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Updates version string in pyproject.toml."
    )
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()

    output = subprocess.check_output(["git", "describe"], encoding="utf-8").rstrip()
    print(f"git describe: {output}")

    m = re.search(r"^v(\d+)\.(\d+)\.(\d+)(-(\d+)-g([a-z0-9]+))?", output)
    major = m.group(1)
    minor = m.group(2)
    patch = m.group(3)
    commits_since_tag = m.group(5)
    git_hash = m.group(6)
    if commits_since_tag:
        version = f"{major}.{minor}.{int(patch) + 1}.dev{commits_since_tag}"
        if not args.release:
            version += f"+g{git_hash}"
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
