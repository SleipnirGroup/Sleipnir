#!/bin/bash
rm -f cachegrind.out.*
valgrind --tool=cachegrind --cache-sim=yes $@
