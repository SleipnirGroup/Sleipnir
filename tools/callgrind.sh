#!/bin/bash
rm -f callgrind.out.*
valgrind --tool=callgrind $@
