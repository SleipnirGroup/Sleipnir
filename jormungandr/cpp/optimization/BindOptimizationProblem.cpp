// Copyright (c) Sleipnir contributors

#include "optimization/BindOptimizationProblem.hpp"

#include <fmt/core.h>
#include <pybind11/pytypes.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>

#include "sleipnir/optimization/SolverConfig.hpp"

namespace py = pybind11;

namespace sleipnir {

void BindOptimizationProblem(py::module_& optimization) {
  using namespace pybind11::literals;

  py::class_<OptimizationProblem> cls{optimization, "OptimizationProblem"};
  cls.def(py::init<>());
  cls.def("decision_variable",
          py::overload_cast<>(&OptimizationProblem::DecisionVariable));
  cls.def("decision_variable",
          py::overload_cast<int, int>(&OptimizationProblem::DecisionVariable),
          "rows"_a, "cols"_a = 1);
  cls.def("symmetric_decision_variable",
          &OptimizationProblem::SymmetricDecisionVariable, "rows"_a);
  cls.def("minimize",
          py::overload_cast<const Variable&>(&OptimizationProblem::Minimize));
  cls.def("minimize", [](OptimizationProblem& self,
                         const VariableMatrix& cost) { self.Minimize(cost); });
  cls.def("minimize",
          [](OptimizationProblem& self, double cost) { self.Minimize(cost); });
  cls.def("maximize",
          py::overload_cast<const Variable&>(&OptimizationProblem::Maximize));
  cls.def("maximize",
          [](OptimizationProblem& self, const VariableMatrix& objective) {
            self.Maximize(objective);
          });
  cls.def("maximize", [](OptimizationProblem& self, double objective) {
    self.Maximize(objective);
  });
  cls.def("subject_to", py::overload_cast<const EqualityConstraints&>(
                            &OptimizationProblem::SubjectTo));
  cls.def("subject_to", py::overload_cast<const InequalityConstraints&>(
                            &OptimizationProblem::SubjectTo));
  cls.def("solve", [](OptimizationProblem& self, const py::kwargs& kwargs) {
    SolverConfig config;
    for (auto& [key, value] : kwargs) {
      auto key_str = key.cast<std::string>();
      if (key_str == "tolerance") {
        config.tolerance = value.cast<double>();
      } else if (key_str == "max_iterations") {
        config.maxIterations = value.cast<int>();
      } else if (key_str == "timeout") {
        config.timeout = std::chrono::duration<double>{value.cast<double>()};
      } else if (key_str == "diagnostics") {
        config.diagnostics = value.cast<bool>();
      } else if (key_str == "spy") {
        config.spy = value.cast<bool>();
      } else {
        throw py::key_error(
            fmt::format("Invalid keyword argument: {}", key_str));
      }
    }

    return self.Solve(config);
  });
}

}  // namespace sleipnir
