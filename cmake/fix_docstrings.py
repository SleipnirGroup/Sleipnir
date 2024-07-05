#!/usr/bin/env python3

"""Fixes errors in Docstrings.hpp."""

import re
import sys


def main():
    # sys.argv[1] should be the filepath of Docstrings.hpp
    filename = sys.argv[1]

    with open(filename) as f:
        content = f.read()

    # Convert parameter names from camel case to snake case
    new_content = ""
    extract_location = 0
    for match in re.finditer(r"(?<=Parameter ``)(.*?)(?=``:)", content):
        new_content += content[extract_location : match.start()]
        param = match.group()
        for i in range(len(param)):
            # Replace uppercase letter preceded by lowercase letter with
            # underscore and lowercase version of letter
            if i > 0 and param[i - 1].islower() and param[i].isupper():
                new_content += "_" + param[i].lower()
            else:
                new_content += param[i]
        extract_location = match.end()
    content = new_content + content[extract_location:]

    with open(filename, mode="w") as f:
        f.write(content)


if __name__ == "__main__":
    main()
