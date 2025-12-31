#!/bin/bash
set -e

cmake --build build -t _sleipnir
cp build/_sleipnir.abi3.so python/src/sleipnir
PYTHONPATH=python/src pytest

rm python/src/sleipnir/_sleipnir.abi3.so
