#!/bin/bash

# Run pytest on locally built Sleipnir extension module

if ! cmake --build build -t _sleipnir; then
  exit 1
fi
cp build/_sleipnir.abi3.so python/src/sleipnir
if PYTHONPATH=python/src pytest "$@"; then
  rm python/src/sleipnir/_sleipnir.abi3.so
  exit 0
else
  rm python/src/sleipnir/_sleipnir.abi3.so
  exit 1
fi
