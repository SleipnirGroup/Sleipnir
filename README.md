# Sleipnir

![C++](https://github.com/SleipnirGroup/Sleipnir/actions/workflows/cpp.yml/badge.svg)
![Python](https://github.com/SleipnirGroup/Sleipnir/actions/workflows/python.yml/badge.svg)
[![PyPI Downloads](https://img.shields.io/pypi/dm/sleipnirgroup-jormungandr.svg?label=PyPI%20Downloads)](https://pypi.org/project/sleipnirgroup-jormungandr/)
[![Website](https://img.shields.io/website?url=https%3A%2F%2Fsleipnirgroup.github.io%2FSleipnir%2F&label=Website)](https://sleipnirgroup.github.io/Sleipnir/)
[![C++ API](https://img.shields.io/badge/documentation-C%2B%2B-blue?label=API%20Docs)](https://sleipnirgroup.github.io/Sleipnir/docs/cpp)
[![Python API](https://img.shields.io/badge/documentation-Python-blue?label=API%20Docs)](https://sleipnirgroup.github.io/Sleipnir/docs/py)
[![Discord](https://img.shields.io/discord/975739302933856277?color=%23738ADB&label=Join%20our%20Discord&logo=discord&logoColor=white)](https://discord.gg/ad2EEZZwsS)

> Sparsity and Linearity-Exploiting Interior-Point solver - Now Internally Readable

Named after Odin's eight-legged horse from Norse mythology, Sleipnir is a reverse mode autodiff library, interior-point method, and NLP solver DSL for C++23 and Python. The DSL automatically chooses the best solver based on the problem structure.

```cpp
#include <print>

#include <sleipnir/optimization/problem.hpp>

int main() {
  // Find the x, y pair with the largest product for which x + 3y = 36
  slp::Problem<double> problem;

  auto x = problem.decision_variable();
  auto y = problem.decision_variable();

  problem.maximize(x * y);
  problem.subject_to(x + 3 * y == 36);
  problem.solve();

  // x = 18.0, y = 6.0
  std::println("x = {}, y = {}", x.value(), y.value());
}
```

```python
#!/usr/bin/env python3

from sleipnir.optimization import Problem


def main():
    # Find the x, y pair with the largest product for which x + 3y = 36
    problem = Problem()

    x, y = problem.decision_variable(2)

    problem.maximize(x * y)
    problem.subject_to(x + 3 * y == 36)
    problem.solve()

    # x = 18.0, y = 6.0
    print(f"x = {x.value()}, y = {y.value()}")


if __name__ == "__main__":
    main()
```

Here's the Python output with `problem.solve(diagnostics=True)`.
```bash
User-configured exit conditions:
  ↳ error below 1e-08
  ↳ iteration callback requested stop
  ↳ executed 5000 iterations

Problem structure:
  ↳ quadratic cost function
  ↳ linear equality constraints
  ↳ no inequality constraints

2 decision variables
1 equality constraint
  ↳ 1 linear
0 inequality constraints

Invoking SQP solver

┏━━━━┯━━━━┯━━━━━━━━━┯━━━━━━━━━━━━┯━━━━━━━━━━━━━┯━━━━━━━━━━━━┯━━━━━━━━━━━━┯━━━━━━━━┯━━━━━┯━━━━━━━━┯━━━━━━━━┯━━┓
┃iter│type│time (ms)│   error    │    cost     │  infeas.   │complement. │   μ    │ reg │primal α│ dual α │↩ ┃
┡━━━━┷━━━━┷━━━━━━━━━┷━━━━━━━━━━━━┷━━━━━━━━━━━━━┷━━━━━━━━━━━━┷━━━━━━━━━━━━┷━━━━━━━━┷━━━━━┷━━━━━━━━┷━━━━━━━━┷━━┩
│   0 norm     0.006 1.799760e-03 -1.080000e+02 6.016734e-10 0.000000e+00 0.00e+00 10⁻⁴  1.00e+00 1.00e+00  0│
│   1 norm     0.008 1.199700e-07 -1.080000e+02 9.947598e-14 0.000000e+00 0.00e+00 10⁻⁴  1.00e+00 1.00e+00  0│
│   2 norm     0.002 4.998668e-12 -1.080000e+02 0.000000e+00 0.000000e+00 0.00e+00 10⁻⁴  1.00e+00 1.00e+00  0│
└────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
┏━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━┯━━━━━━━━━┯━━━━┓
┃     solver trace      │     percent      │total (ms)│each (ms)│runs┃
┡━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━┷━━━━━━━━━┷━━━━┩
│solver                  100.00%▕█████████▏      0.056     0.056    1│
│  ↳ setup                 5.36%▕▍        ▏      0.003     0.003    1│
│  ↳ iteration            30.36%▕██▋      ▏      0.017     0.005    3│
│    ↳ feasibility ✓       0.00%▕         ▏      0.000     0.000    3│
│    ↳ iter callbacks      0.00%▕         ▏      0.000     0.000    3│
│    ↳ KKT matrix build    1.79%▕▏        ▏      0.001     0.000    3│
│    ↳ KKT matrix decomp  14.29%▕█▎       ▏      0.008     0.002    3│
│    ↳ KKT system solve    1.79%▕▏        ▏      0.001     0.000    3│
│    ↳ line search         1.79%▕▏        ▏      0.001     0.000    3│
│      ↳ SOC               0.00%▕         ▏      0.000     0.000    0│
│    ↳ next iter prep      0.00%▕         ▏      0.000     0.000    3│
│    ↳ f(x)                0.00%▕         ▏      0.000     0.000    7│
│    ↳ ∇f(x)               1.79%▕▏        ▏      0.001     0.000    4│
│    ↳ ∇²ₓₓL               0.00%▕         ▏      0.000     0.000    4│
│    ↳ cₑ(x)               1.79%▕▏        ▏      0.001     0.000    7│
│    ↳ ∂cₑ/∂x              0.00%▕         ▏      0.000     0.000    4│
└────────────────────────────────────────────────────────────────────┘
┏━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━┯━━━━━━━━━┯━━━━┓
┃    autodiff trace     │     percent      │total (ms)│each (ms)│runs┃
┡━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━┷━━━━━━━━━┷━━━━┩
│setup                   100.00%▕█████████▏      0.013     0.013    1│
│  ↳ ∇f(x)                 7.69%▕▋        ▏      0.001     0.001    1│
│  ↳ ∂cₑ/∂x                7.69%▕▋        ▏      0.001     0.001    1│
│  ↳ ∇²ₓₓL                38.46%▕███▍     ▏      0.005     0.005    1│
└────────────────────────────────────────────────────────────────────┘

Exit: success
x = 17.99999999999167, y = 6.0000000000027764
```

The C++ API also supports arbitrary scalar types, so users can specify higher precision floating-point types at the cost of speed.

Sleipnir's internals are intended to be readable by those who aren't domain experts with links to explanatory material for its algorithms.

## Benchmarks

<table><tr>
  <td><img src="flywheel-scalability-results.png" alt="flywheel-scalability-results"/></td>
  <td><img src="cart-pole-scalability-results.png" alt="cart-pole-scalability-results"/></td>
</tr><tr>
  <td>
    <a href="flywheel-scalability-results-casadi.csv">
      flywheel-scalability-results-casadi.csv
    </a><br>
    <a href="flywheel-scalability-results-sleipnir.csv">
      flywheel-scalability-results-sleipnir.csv
    </a>
  </td>
  <td>
    <a href="cart-pole-scalability-results-casadi.csv">
      cart-pole-scalability-results-casadi.csv
    </a><br>
    <a href="cart-pole-scalability-results-sleipnir.csv">
      cart-pole-scalability-results-sleipnir.csv
    </a>
  </td>
</tr></table>

Generated by [tools/generate-scalability-results.sh](https://github.com/SleipnirGroup/Sleipnir/tree/main/tools/generate-scalability-results.sh) from [benchmarks/scalability](https://github.com/SleipnirGroup/Sleipnir/tree/main/benchmarks/scalability) source.

* CPU: AMD Ryzen 7 7840U
* RAM: 64 GB, 5600 MHz DDR5
* Compiler version: g++ (GCC) 15.2.1 20250813

The following thirdparty software was used in the benchmarks:

* CasADi 3.7.2 (autodiff and NLP solver frontend)
* Ipopt 3.14.19 (NLP solver backend)
* MUMPS 5.7.3 (linear solver)

Ipopt uses MUMPS by default because it has free licensing. Commercial linear solvers may be much faster.

See [benchmark details](https://github.com/SleipnirGroup/Sleipnir/?tab=readme-ov-file#benchmark-details) for more.

## Install

### Minimum system requirements

* Windows
  * OS: Windows 11
  * Runtime: [Microsoft Visual C++ 2022 redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) from Visual Studio 2022 17.13
* Linux
  * OS: Ubuntu 24.04
  * Runtime: GCC 14 libstdc++ (run `sudo apt install g++-14`)
* macOS
  * OS: macOS 14.5
  * Runtime: Apple Clang 16.0.0 libc++ from Xcode 16.2 (run `xcode-select --install`)

### C++ library

To install Sleipnir system-wide, see the [build instructions](https://github.com/SleipnirGroup/Sleipnir/?tab=readme-ov-file#c-library-1).

To use Sleipnir within a CMake project, add the following to your CMakeLists.txt:
```cmake
include(FetchContent)

FetchContent_Declare(
    Sleipnir
    GIT_REPOSITORY https://github.com/SleipnirGroup/Sleipnir.git
    GIT_TAG main
    EXCLUDE_FROM_ALL
    SYSTEM
)
FetchContent_MakeAvailable(Sleipnir)

target_link_libraries(MyApp PUBLIC Sleipnir::Sleipnir)
```

### Python library

```bash
pip install sleipnirgroup-jormungandr
```

## API docs

See the [C++ API docs](https://sleipnirgroup.github.io/Sleipnir/docs/cpp) and [Python API docs](https://sleipnirgroup.github.io/Sleipnir/docs/py).

## Examples

See the [examples](https://github.com/SleipnirGroup/Sleipnir/tree/main/examples), [C++ optimization unit tests](https://github.com/SleipnirGroup/Sleipnir/tree/main/test/optimization), and [Python optimization unit tests](https://github.com/SleipnirGroup/Sleipnir/tree/main/python/test/optimization).

## Build

### Dependencies

* C++23 compiler
  * On Windows 11 or greater, install [Visual Studio Community 2022](https://visualstudio.microsoft.com/vs/community/) and select the C++ programming language during installation
  * On Ubuntu 24.04 or greater, install GCC 14 via `sudo apt install g++-14`
  * On macOS 14.5 or greater, install the Xcode 16.2 command-line build tools via `xcode-select --install`
* [CMake](https://cmake.org/download/) 3.21 or greater
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

### C++ library

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
* Perf
  * RelWithDebInfo build type, but with frame pointer so perf utility can use it

### Python library

On Windows, open a [Developer PowerShell](https://learn.microsoft.com/en-us/visualstudio/ide/reference/command-prompt-powershell?view=vs-2022). On Linux or macOS, open a Bash shell.

```bash
# Clone the repository
git clone git@github.com:SleipnirGroup/Sleipnir
cd Sleipnir

# Setup
pip install --user build

# Build
python -m build --wheel

# Install
pip install --user dist/sleipnirgroup_jormungandr-*.whl

# Test
pytest
```

## Test diagnostics

Passing the `--enable-diagnostics` flag to the test executable enables solver diagnostic prints.

Some test problems generate CSV files containing their solutions. These can be plotted with [tools/plot_test_problem_solutions.py](https://github.com/SleipnirGroup/Sleipnir/blob/main/tools/plot_test_problem_solutions.py).

## Benchmark details

### Running the benchmarks

Benchmark projects are in the [benchmarks folder](https://github.com/SleipnirGroup/Sleipnir/tree/main/benchmarks). To compile and run them, run the following in the repository root:
```bash
# Install CasADi and [matplotlib, numpy, scipy] pip packages first
cmake -B build -S . -DBUILD_BENCHMARKS=ON
cmake --build build
./tools/generate-scalability-results.sh
```

See the contents of `./tools/generate-scalability-results.sh` for how to run specific benchmarks.

### How we improved performance

#### Make more decisions at compile time

During problem setup, equality and inequality constraints are encoded as different types, so the appropriate setup behavior can be selected at compile time via operator overloads.

#### Reuse autodiff computation results that are still valid (aka caching)

The autodiff library automatically records the linearity of every node in the computational graph. Linear functions have constant first derivatives, and quadratic functions have constant second derivatives. The constant derivatives are computed in the initialization phase and reused for all solver iterations. Only nonlinear parts of the computational graph are recomputed during each solver iteration.

For quadratic problems, we compute the Lagrangian Hessian and constraint Jacobians once with no problem structure hints from the user.

#### Use a performant linear algebra library with fast sparse solvers

[Eigen](https://gitlab.com/libeigen/eigen) provides these. It also has no required dependencies, which makes cross compilation much easier.

#### Use a pool allocator for autodiff expression nodes

This promotes fast allocation/deallocation and good memory locality.

We could mitigate the solver's high last-level-cache miss rate (~42% on the machine above) further by breaking apart the expression nodes into fields that are commonly iterated together. We used to use a tape, which gave computational graph updates linear access patterns, but tapes are monotonic buffers with no way to reclaim storage.
