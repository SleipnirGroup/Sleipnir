# Sleipnir

> Sparse Linearity-Exploiting Interior-Point solver - Now Internally Readable

Sleipnir is a linearity-exploiting sparse nonlinear optimization problem solver that uses the interior-point method and has readable internals, unlike other optimization problem solvers. It's named after Odin's eight-legged horse from Norse mythology.

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

To compile and run the flywheel scalability benchmark, run:
```bash
# Install CasADi first
cmake -B build -S .
cmake --build build
./build/FlywheelScalabilityBenchmark
```

This will generate a scalability-results.csv in the current directory. To plot the results, run:
```bash
# Install matplotlib, numpy, and scipy pip packages first
./tools/plot_scalability_results.py
```

## Algorithm documentation

### Reverse accumulation automatic differentiation

In reverse accumulation AD, the dependent variable to be differentiated is fixed and the derivative is computed with respect to each subexpression recursively. In a pen-and-paper calculation, the derivative of the outer functions is repeatedly substituted in the chain rule:

(∂y/∂x) = (∂y/∂w₁) ⋅ (∂w₁/∂x) = ((∂y/∂w₂) ⋅ (∂w₂/∂w₁)) ⋅ (∂w₁/∂x) = ...

In reverse accumulation, the quantity of interest is the adjoint, denoted with a bar (w̄); it is a derivative of a chosen dependent variable with respect to a subexpression w: ∂y/∂w.

Given the expression f(x₁,x₂)=sin(x₁) + x₁x₂, the computational graph is:
```
               f(x₁,x₂)
                  |
                  w₅     w₅=w₄+w₃
                 /  \
                /    \
 w₄=sin(w₁)    w₄     w₃    w₃=w₁w₂
               |   /  |
     w₁=x₁     w₁     w₂    w₂=x₃
```

The operations to compute the derivative:

w̄₅ = 1 (seed)\
w̄₄ = w̄₅(∂w₅/∂w₄) = w̄₅\
w̄₃ = w̄₅(∂w₅/∂w₃) = w̄₅\
w̄₂ = w̄₃(∂w₃/∂w₂) = w̄₃w₁\
w̄₁ = w̄₄(∂w₄/∂w₁) + w̄₃(∂w₃/∂w₁) = w̄₄cos(w₁) + w̄₃w₂

https://en.wikipedia.org/wiki/Automatic_differentiation#Beyond_forward_and_reverse_accumulation
