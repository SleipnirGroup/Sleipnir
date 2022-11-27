perf stat -B -e cache-references,cache-misses,LLC-load-misses,LLC-loads,LLC-store-misses,LLC-stores,cycles,instructions,branches,faults,migrations $@
