#!/usr/bin/env python

"""Prints extra pip package constraints for CI build."""

import re
import subprocess

# Constrain clang package to locally installed major version for ABI compat
output = subprocess.check_output(["clang++", "--version"], encoding="utf-8")
print(f"clang~={re.search(r'[0-9]+', output).group()}.0")
