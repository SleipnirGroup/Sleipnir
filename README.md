# Sleipnir

![Build](https://github.com/calcmogul/Sleipnir/actions/workflows/build.yml/badge.svg)
[![C++ Documentation](https://img.shields.io/badge/documentation-c%2B%2B-blue)](https://calcmogul.github.io/Sleipnir/)

> Sparsity and Linearity-Exploiting Interior-Point solver - Now Internals are Readable

Named after Odin's eight-legged horse from Norse mythology, Sleipnir is a linearity-exploiting sparse nonlinear constrained optimization problem solver that uses the interior-point method.

Sleipnir's internals are intended to be readable by those who aren't domain experts, unlike other optimization problem solvers.

## Examples

See the [examples](https://github.com/calcmogul/Sleipnir/tree/main/examples) and [optimization unit tests](https://github.com/calcmogul/Sleipnir/tree/main/test/optimization).

## Dependencies

* C++20 compiler
  * On Linux, install GCC 11 or greater
  * On Windows, install [Visual Studio Community 2022](https://visualstudio.microsoft.com/vs/community/) and select the C++ programming language during installation
  * On macOS, install the Xcode command-line build tools via `xcode-select --install`
* [Eigen](https://gitlab.com/libeigen/eigen)
* [fmtlib](https://github.com/fmtlib/fmt) (internal only)
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

### Supported build types

The following build types can be specified via `-DCMAKE_BUILD_TYPE`:

* Debug
  * Optimizations off
  * Debug symbols on
* Release
  * Optimizations on
  * Debug symbols off
* RelWithDebInfo (default)
  * Release build type, but with debug info
* MinSizeRel
  * Minimum size release build
* Asan
  * Enables address sanitizer
* Tsan
  * Enables thread sanitizer
* Ubsan
  * Enables undefined behavior sanitizer
* Perf
  * RelWithDebInfo build type, but with frame pointer so perf utility can use it

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
