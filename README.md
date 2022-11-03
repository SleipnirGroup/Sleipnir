# Sleipnir

> Sparsity and Linearity-Exploiting Interior-Point solver - Now Internals are Readable

Named after Odin's eight-legged horse from Norse mythology, Sleipnir is a linearity-exploiting sparse nonlinear constrained optimization problem solver that uses the interior-point method.

Sleipnir's internals are intended to be readable by those who aren't domain experts, unlike other optimization problem solvers.

## Documentation

Run the following in the repository root:
```bash
mkdir build
doxygen docs/Doxyfile
```

Open `build/docs/html/index.html` in a browser.

## Examples

See the [examples](examples) and [optimization unit tests](test/optimization).

## Dependencies

* C++20 compiler
  * On Linux, install GCC 11 or greater
  * On Windows, install [Visual Studio Community 2022](https://visualstudio.microsoft.com/vs/community/) and select the C++ programming language during installation
  * On macOS, install the Xcode command-line build tools via `xcode-select --install`
* [Eigen](https://gitlab.com/libeigen/eigen)
* [fmtlib](https://github.com/fmtlib/fmt)
* [googletest](https://github.com/google/googletest) (tests only)

Library dependencies which aren't installed locally will be automatically downloaded and built by CMake.

If [CasADi](https://github.com/casadi/casadi) is installed locally, the benchmark executables will be built.

## Build instructions

Starting from the repository root, run the configure step:
```bash
cmake -B build -S .
```

This will automatically download library dependencies.

Run the build step:
```bash
cmake --build build
```

Run the tests:
```bash
cd build
ctest -R SleipnirTest
```

A regex is used to filter for the Sleipnir tests because CMake includes the tests of dependencies in our test set; they take a long time and are unnecessary.

## Benchmarks

The following benchmarks are available:

* benchmarks/scalability/cartpole
* benchmarks/scalability/flywheel

To compile and run the flywheel scalability benchmark, run the following in the repository root:
```bash
# Install CasADi first
cmake -B build -S .
cmake --build build
./build/FlywheelScalabilityBenchmark
```

This will generate a scalability-results.csv. To plot the results, run the following:
```bash
# Install matplotlib, numpy, and scipy pip packages first
./tools/plot_scalability_results.py
```
