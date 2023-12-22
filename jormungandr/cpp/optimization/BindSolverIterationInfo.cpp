// Copyright (c) Sleipnir contributors

#include <pybind11/eigen.h>
#include <sleipnir/optimization/SolverIterationInfo.hpp>

#include "optimization/BindSolverStatus.hpp"

namespace py = pybind11;

namespace sleipnir {

void BindSolverIterationInfo(py::module_& optimization) {
  py::class_<SolverIterationInfo> cls{optimization, "SolverIterationInfo"};
  cls.def_readonly("iteration", &SolverIterationInfo::iteration);
  cls.def_property_readonly(
      "x", [](const SolverIterationInfo& self) { return self.x; });
  cls.def_property_readonly("g", [](const SolverIterationInfo& self) {
    return Eigen::SparseMatrix<double>{self.g};
  });
  cls.def_property_readonly(
      "H", [](const SolverIterationInfo& self) { return self.H; });
  cls.def_property_readonly(
      "A_e", [](const SolverIterationInfo& self) { return self.A_e; });
  cls.def_property_readonly(
      "A_i", [](const SolverIterationInfo& self) { return self.A_i; });
}

}  // namespace sleipnir
