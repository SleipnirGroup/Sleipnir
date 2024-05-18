// Copyright (c) Sleipnir contributors

#include <format>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <sleipnir/optimization/SolverConfig.hpp>

#include "Docstrings.hpp"

namespace py = pybind11;

namespace sleipnir {

void BindOptimizationProblem(py::class_<OptimizationProblem>& cls) {
  using namespace py::literals;

  cls.def(py::init<>(),
          DOC(sleipnir, OptimizationProblem, OptimizationProblem));
  cls.def("decision_variable",
          py::overload_cast<>(&OptimizationProblem::DecisionVariable),
          DOC(sleipnir, OptimizationProblem, DecisionVariable));
  cls.def("decision_variable",
          py::overload_cast<int, int>(&OptimizationProblem::DecisionVariable),
          "rows"_a, "cols"_a = 1,
          DOC(sleipnir, OptimizationProblem, DecisionVariable, 2));
  cls.def("symmetric_decision_variable",
          &OptimizationProblem::SymmetricDecisionVariable, "rows"_a,
          DOC(sleipnir, OptimizationProblem, SymmetricDecisionVariable));
  cls.def("minimize",
          py::overload_cast<const Variable&>(&OptimizationProblem::Minimize),
          "cost"_a, DOC(sleipnir, OptimizationProblem, Minimize));
  cls.def(
      "minimize",
      [](OptimizationProblem& self, const VariableMatrix& cost) {
        self.Minimize(cost);
      },
      "cost"_a, DOC(sleipnir, OptimizationProblem, Minimize));
  cls.def(
      "minimize",
      [](OptimizationProblem& self, double cost) { self.Minimize(cost); },
      "cost"_a, DOC(sleipnir, OptimizationProblem, Minimize));
  cls.def("maximize",
          py::overload_cast<const Variable&>(&OptimizationProblem::Maximize),
          "objective"_a, DOC(sleipnir, OptimizationProblem, Maximize));
  cls.def(
      "maximize",
      [](OptimizationProblem& self, const VariableMatrix& objective) {
        self.Maximize(objective);
      },
      "objective"_a, DOC(sleipnir, OptimizationProblem, Maximize));
  cls.def(
      "maximize",
      [](OptimizationProblem& self, double objective) {
        self.Maximize(objective);
      },
      "objective"_a, DOC(sleipnir, OptimizationProblem, Maximize));
  cls.def("subject_to",
          py::overload_cast<const EqualityConstraints&>(
              &OptimizationProblem::SubjectTo),
          "constraint"_a, DOC(sleipnir, OptimizationProblem, SubjectTo));
  cls.def("subject_to",
          py::overload_cast<const InequalityConstraints&>(
              &OptimizationProblem::SubjectTo),
          "constraint"_a, DOC(sleipnir, OptimizationProblem, SubjectTo, 3));
  cls.def(
      "solve",
      [](OptimizationProblem& self, const py::kwargs& kwargs) {
        SolverConfig config;
        for (auto& [key, value] : kwargs) {
          auto key_str = key.cast<std::string>();
          if (key_str == "tolerance") {
            config.tolerance = value.cast<double>();
          } else if (key_str == "max_iterations") {
            config.maxIterations = value.cast<int>();
          } else if (key_str == "timeout") {
            config.timeout =
                std::chrono::duration<double>{value.cast<double>()};
          } else if (key_str == "diagnostics") {
            config.diagnostics = value.cast<bool>();
          } else if (key_str == "spy") {
            config.spy = value.cast<bool>();
          } else {
            throw py::key_error(
                std::format("Invalid keyword argument: {}", key_str));
          }
        }

        return self.Solve(config);
      },
      DOC(sleipnir, OptimizationProblem, Solve));
  cls.def(
      "callback",
      [](OptimizationProblem& self,
         std::function<bool(const SolverIterationInfo&)> callback) {
        self.Callback(std::move(callback));
      },
      "callback"_a, DOC(sleipnir, OptimizationProblem, Callback, 2));
}

}  // namespace sleipnir
