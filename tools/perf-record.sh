#!/bin/bash
perf record --call-graph fp -- "$@"
