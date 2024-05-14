#!/usr/bin/env python3

from current_manager import CurrentManager


def main():
    manager = CurrentManager([1.0, 5.0, 10.0, 5.0], 40.0)
    currents = manager.calculate([25.0, 10.0, 5.0, 0.0])
    print(f"currents = {currents}")


if __name__ == "__main__":
    main()
