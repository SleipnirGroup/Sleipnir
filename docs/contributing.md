# Contributing

## Building Sleipnir

### Dependencies

* C++23 compiler
  * On Windows 11 or greater, install [Visual Studio Community 2022](https://visualstudio.microsoft.com/vs/community/) and select the C++ programming language during installation
  * On Ubuntu 24.04 or greater, install GCC 14 via `sudo apt install g++-14`
  * On macOS 14.5 or greater, install the Xcode 16.2 command-line build tools via `xcode-select --install`
* [CMake](https://cmake.org/download/) 3.28 or greater
  * On Windows, install from the link above
  * On Linux, install via `sudo apt install cmake`
  * On macOS, install via `brew install cmake`
* [Python](https://www.python.org/downloads/) 3.12 or greater
  * On Windows, install from the link above
  * On Linux, install via `sudo apt install python`
  * On macOS, install via `brew install python`
* [Eigen](https://gitlab.com/libeigen/eigen)
* [small_vector](https://github.com/gharveymn/small_vector)
* [nanobind](https://github.com/wjakob/nanobind) (build only)
* [Catch2](https://github.com/catchorg/Catch2) (tests only)

Library dependencies which aren't installed locally will be automatically downloaded and built by CMake.

The benchmark executables require [CasADi](https://github.com/casadi/casadi) to be installed locally.

#### C++ library

On Windows, open a [Developer PowerShell](https://learn.microsoft.com/en-us/visualstudio/ide/reference/command-prompt-powershell?view=vs-2022). On Linux or macOS, open a Bash shell.

```bash
# Clone the repository
git clone git@github.com:SleipnirGroup/Sleipnir
cd Sleipnir

# Configure; automatically downloads library dependencies
cmake -B build -S .

# Build
cmake --build build

# Test
ctest --test-dir build --output-on-failure

# Install
cmake --install build --prefix pkgdir
```

The following build types can be specified via `-DCMAKE_BUILD_TYPE` during CMake configure:

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
* Coverage
  * Enables code coverage instrumentation: source-based coverage with Clang, gcov-based coverage with GCC (see `tools/generate_coverage.py`)

##### CMake presets

`CMakePresets.json` provides presets for common build configurations. Each preset uses the Ninja generator and its own build directory, so multiple configurations can coexist. Pass the same preset name to each step.

```bash
# Configure
cmake --preset asan

# Build
cmake --build --preset asan

# Test
ctest --preset asan
```

Run `cmake --list-presets=all` to list the available presets.

| Preset                | Build directory             | Description                           |
|-----------------------|-----------------------------|---------------------------------------|
| `default`             | `build`                     | RelWithDebInfo with benchmarks        |
| `debug`               | `build-debug`               | Debug                                 |
| `bound-projection`    | `build-bound-projection`    | `SLEIPNIR_ENABLE_BOUND_PROJECTION=ON` |
| `disable-diagnostics` | `build-disable-diagnostics` | `SLEIPNIR_DISABLE_DIAGNOSTICS=ON`     |
| `examples`            | `build-examples`            | Examples                              |
| `python`              | `build-python`              | Python module                         |
| `asan`                | `build-asan`                | Address sanitizer                     |
| `tsan`                | `build-tsan`                | Thread sanitizer                      |
| `ubsan`               | `build-ubsan`               | Undefined behavior sanitizer          |
| `coverage-clang`      | `build-coverage-clang`      | Clang source-based code coverage      |
| `coverage-gcc`        | `build-coverage-gcc`        | GCC gcov-based code coverage          |
| `aarch64-linux-gnu`   | `build-aarch64-linux-gnu`   | Cross-compile for aarch64 Linux       |
| `arm-none-eabi`       | `build-arm-none-eabi`       | Cross-compile for bare-metal ARM      |

The `aarch64-linux-gnu` and `arm-none-eabi` presets have no test preset. The presets build only the `double` test variants; pass `-DSLEIPNIR_TEST_SCALAR_TYPES="double;ExplicitDouble"` during configure to also build the `ExplicitDouble` variants like CI does.

The `coverage-clang` test preset writes raw profiles to `build-coverage-clang/ctest-*.profraw` (one per test executable), which can be merged with `llvm-profdata merge` and reported with `llvm-cov`. The `coverage-gcc` test preset writes gcov counters to `.gcda` files next to the object files, which can be reported with `gcovr`. Both presets accumulate counters across runs, and rebuilding the Clang test executables leaves stale `.profraw` files behind, so delete the `.profraw` or `.gcda` files before rerunning. To generate HTML and text reports for specific test executables instead, run `tools/generate_coverage.py`, which builds in a separate `build-coverage` directory.

To use a compiler cache like ccache, set the `CMAKE_C_COMPILER_LAUNCHER` and `CMAKE_CXX_COMPILER_LAUNCHER` environment variables before configuring. Personal presets can be added in `CMakeUserPresets.json`, which is ignored by Git.

#### Python library

On Windows, open a [Developer PowerShell](https://learn.microsoft.com/en-us/visualstudio/ide/reference/command-prompt-powershell?view=vs-2022). On Linux or macOS, open a Bash shell.

```bash
# Clone the repository
git clone git@github.com:SleipnirGroup/Sleipnir
cd Sleipnir

# Setup
pip install --user build

# Build
./tools/update_version.py
python -m build --wheel

# Install
pip install --user dist/sleipnirgroup_jormungandr-*.whl

# Test
pytest
```

When configuring CMake directly with `-DSLEIPNIR_BUILD_PYTHON=ON` instead of building a wheel, install pybind11_mkdoc first via `pip install --group docstrings`. It regenerates the Python docstrings from the C++ headers.

## Development

### Running C++ tests

The following runs the C++ tests on a development build.

```bash
cmake --build build
ctest --test-dir build --output-on-failure
```

### Running Python tests

The following runs the Python tests on a development build.

```bash
uv run pytest
```

### Running Python examples

The following runs an example on a development build.

```bash
cd examples/...
uv run main.py
```

### Running formatters and linters

The following runs the formatters and linters on the whole repo.

```bash
uvx wpiformat
```

Use the `-f` flag to run wpiformat on a file or directory.

## Educational resources

### Autodiff

Here's an introductory video on autodiff.

> Ari Seff. What is Automatic Differentation?. 2020. https://www.youtube.com/watch?v=wG_nF1awSSY

### Linear algebra

3Blue1Brown's _Essence of Linear Algebra_ video series provides geometric intuition for linear algebra.

> 3Blue1Brown. _Essence of Linear Algebra_, 2016. https://www.3blue1brown.com/topics/linear-algebra

### Convex optimization

Visually Explained's _Convex Optimization_ video series provides geometric intuition for convex optimization and the interior-point method.

> Visually Explained. _Convex Optimization_, 2021. https://www.youtube.com/playlist?list=PLqwozWPBo-FuPu4d9pFOobsCF1vDGdY_I

Sleipnir's authors learned numerical optimization from the following book and relied heavily upon it to implement Sleipnir.

> Nocedal, J. and Wright, S. _Numerical Optimization_, 2nd. ed., Springer, 2006. https://www.math.kent.edu/~reichel/courses/optimization/Numerical_Optimization.pdf

See [this page](https://sleipnirgroup.github.io/Sleipnir/md_algorithms.html) for derivations of Newton's method (unconstrained optimization), sequential quadratic programming (optimization with equality constraints), and interior-point method (optimization with equality and inequality constraints). The bottom of that page lists works cited by Sleipnir's internals.

## Branding

### Logo

[SVG](https://github.com/SleipnirGroup/Sleipnir/tree/main/docs/logo/sleipnir.svg), [PNG (1000px)](https://github.com/SleipnirGroup/Sleipnir/tree/main/docs/logo/sleipnir_THcolors_1000px.png), [PNG (55px)](https://github.com/SleipnirGroup/Sleipnir/tree/main/docs/logo/sleipnir_THcolors_55px.png)<br>
Font: [Centaur](https://en.wikipedia.org/wiki/Centaur_(typeface))

### Color palette

<table>
  <tr>
    <th>Purple</th>
    <th>Yellow</th>
  </tr>
  <tr>
    <td style="background-color: #6d3d94; color: white;">6D3D94</td>
    <td style="background-color: #fdb813; color: white;">FDB813</td>
  </tr>
</table>
