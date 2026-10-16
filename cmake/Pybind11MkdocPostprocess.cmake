# Replaces pybind11_mkdoc's docstring pointers with constexpr arrays
#
# Usage: cmake -DFILE=<docstrings.hpp> -P Pybind11MkdocPostprocess.cmake

file(READ ${FILE} contents)
string(
    REGEX REPLACE "\nstatic const char \\*([A-Za-z0-9_]+) ="
    "\nconstexpr char \\1[] ="
    contents
    "${contents}"
)
file(WRITE ${FILE} "${contents}")
