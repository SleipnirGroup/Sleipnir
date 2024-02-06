#!/usr/bin/env python3

import os
import sys


def main():
    for filename in sys.argv[1:]:
        src = filename
        dest = filename.replace("linux_x86_64", "manylinux_2_35_x86_64")
        if src != dest:
            print(f"{src} -> {dest}")
            os.rename(src, dest)


if __name__ == "__main__":
    main()
