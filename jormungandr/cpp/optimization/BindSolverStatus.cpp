// Copyright (c) Sleipnir contributors

#include "optimization/BindSolverStatus.hpp"

#include <sleipnir/autodiff/Expression.hpp>
#include <sleipnir/optimization/SolverExitCondition.hpp>
#include <sleipnir/optimization/SolverStatus.hpp>

namespace py = pybind11;

namespace sleipnir {

void BindSolverStatus(py::module_& optimization) {
  using namespace pybind11::literals;

  py::class_<SolverStatus> cls{optimization, "SolverStatus"};
  cls.def(py::init<>());
  cls.def(py::init<ExpressionType, ExpressionType, ExpressionType,
                   SolverExitCondition>(),
          "cost_function_type"_a = ExpressionType::kNone,
          "equality_constraint_type"_a = ExpressionType::kNone,
          "inequality_constraint_type"_a = ExpressionType::kNone,
          "exit_condition"_a = SolverExitCondition::kSuccess);
  cls.def_readwrite("cost_function_type", &SolverStatus::costFunctionType);
  cls.def_readwrite("equality_constraint_type",
                    &SolverStatus::equalityConstraintType);
  cls.def_readwrite("inequality_constraint_type",
                    &SolverStatus::inequalityConstraintType);
  cls.def_readwrite("exit_condition", &SolverStatus::exitCondition);
}

}  // namespace sleipnir
