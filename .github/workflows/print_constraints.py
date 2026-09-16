#!/usr/bin/env python

"""Prints extra pip package constraints for CI build."""

import re
import subprocess

# Constrain clang package to locally installed major version for ABI compat,
# limited to what's available on PyPI
output = subprocess.check_output(["clang++", "--version"], encoding="utf-8")
version = int(re.search(r"[0-9]+", output).group())
print(f"clang~={min(version, 21)}.0")
