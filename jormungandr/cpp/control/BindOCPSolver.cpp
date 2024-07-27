// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/function.h>
#include <sleipnir/autodiff/VariableMatrix.hpp>
#include <sleipnir/control/OCPSolver.hpp>

#include "Docstrings.hpp"

namespace nb = nanobind;

namespace sleipnir {

void BindOCPSolver(nb::enum_<TranscriptionMethod>& transcription_method,
                   nb::enum_<DynamicsType>& dynamics_type,
                   nb::enum_<TimestepMethod>& timestep_method,
                   nb::class_<OCPSolver, OptimizationProblem>& cls) {
  using namespace nb::literals;

  transcription_method.value(
      "DIRECT_TRANSCRIPTION", TranscriptionMethod::kDirectTranscription,
      DOC(sleipnir, TranscriptionMethod, kDirectTranscription));
  transcription_method.value(
      "DIRECT_COLLOCATION", TranscriptionMethod::kDirectCollocation,
      DOC(sleipnir, TranscriptionMethod, kDirectCollocation));
  transcription_method.value(
      "SINGLE_SHOOTING", TranscriptionMethod::kSingleShooting,
      DOC(sleipnir, TranscriptionMethod, kSingleShooting));

  dynamics_type.value("EXPLICIT_ODE", DynamicsType::kExplicitODE,
                      DOC(sleipnir, DynamicsType, kExplicitODE));
  dynamics_type.value("DISCRETE", DynamicsType::kDiscrete,
                      DOC(sleipnir, DynamicsType, kDiscrete));

  timestep_method.value("FIXED", TimestepMethod::kFixed,
                        DOC(sleipnir, TimestepMethod, kFixed));
  timestep_method.value("VARIABLE", TimestepMethod::kVariable,
                        DOC(sleipnir, TimestepMethod, kVariable));
  timestep_method.value("VARIABLE_SINGLE", TimestepMethod::kVariableSingle,
                        DOC(sleipnir, TimestepMethod, kVariableSingle));

  cls.def(nb::init<int, int, std::chrono::duration<double>, int,
                   const std::function<VariableMatrix(
                       const VariableMatrix& x, const VariableMatrix& u)>&,
                   DynamicsType, TimestepMethod, TranscriptionMethod>(),
          "num_states"_a, "num_inputs"_a, "dt"_a, "num_steps"_a, "dynamics"_a,
          "dynamics_type"_a = DynamicsType::kExplicitODE,
          "timestep_method"_a = TimestepMethod::kFixed,
          "transcription_method"_a = TranscriptionMethod::kDirectTranscription,
          DOC(sleipnir, OCPSolver, OCPSolver));

  cls.def("constrain_initial_state", &OCPSolver::ConstrainInitialState<double>,
          "initial_state"_a, DOC(sleipnir, OCPSolver, ConstrainInitialState));
  cls.def("constrain_initial_state", &OCPSolver::ConstrainInitialState<int>,
          "initial_state"_a, DOC(sleipnir, OCPSolver, ConstrainInitialState));
  cls.def("constrain_initial_state",
          &OCPSolver::ConstrainInitialState<Variable>, "initial_state"_a,
          DOC(sleipnir, OCPSolver, ConstrainInitialState));
  cls.def(
      "constrain_initial_state",
      [](OCPSolver& self, nb::DRef<Eigen::MatrixXd> initialState) {
        self.ConstrainInitialState(initialState);
      },
      "initial_state"_a, DOC(sleipnir, OCPSolver, ConstrainInitialState));
  cls.def("constrain_initial_state",
          &OCPSolver::ConstrainInitialState<VariableMatrix>, "initial_state"_a,
          DOC(sleipnir, OCPSolver, ConstrainInitialState));

  cls.def("constrain_final_state", &OCPSolver::ConstrainFinalState<double>,
          "final_state"_a, DOC(sleipnir, OCPSolver, ConstrainFinalState));
  cls.def("constrain_final_state", &OCPSolver::ConstrainFinalState<int>,
          "final_state"_a, DOC(sleipnir, OCPSolver, ConstrainFinalState));
  cls.def("constrain_final_state", &OCPSolver::ConstrainFinalState<Variable>,
          "final_state"_a, DOC(sleipnir, OCPSolver, ConstrainFinalState));
  cls.def(
      "constrain_final_state",
      [](OCPSolver& self, nb::DRef<Eigen::MatrixXd> finalState) {
        self.ConstrainFinalState(finalState);
      },
      "final_state"_a, DOC(sleipnir, OCPSolver, ConstrainFinalState));
  cls.def("constrain_final_state",
          &OCPSolver::ConstrainFinalState<VariableMatrix>, "final_state"_a,
          DOC(sleipnir, OCPSolver, ConstrainFinalState));

  cls.def(
      "for_each_step",
      [](OCPSolver& self,
         const std::function<void(const VariableMatrix& x,
                                  const VariableMatrix& u)>& callback) {
        self.ForEachStep(callback);
      },
      "callback"_a, DOC(sleipnir, OCPSolver, ForEachStep));

  cls.def("set_lower_input_bound", &OCPSolver::SetLowerInputBound<double>,
          "lower_bound"_a, DOC(sleipnir, OCPSolver, SetLowerInputBound));
  cls.def("set_lower_input_bound", &OCPSolver::SetLowerInputBound<int>,
          "lower_bound"_a, DOC(sleipnir, OCPSolver, SetLowerInputBound));
  cls.def("set_lower_input_bound", &OCPSolver::SetLowerInputBound<Variable>,
          "lower_bound"_a, DOC(sleipnir, OCPSolver, SetLowerInputBound));
  cls.def(
      "set_lower_input_bound",
      [](OCPSolver& self, nb::DRef<Eigen::MatrixXd> lowerBound) {
        self.SetLowerInputBound(lowerBound);
      },
      "lower_bound"_a, DOC(sleipnir, OCPSolver, SetLowerInputBound));
  cls.def("set_lower_input_bound",
          &OCPSolver::SetLowerInputBound<VariableMatrix>, "lower_bound"_a,
          DOC(sleipnir, OCPSolver, SetLowerInputBound));

  cls.def("set_upper_input_bound", &OCPSolver::SetUpperInputBound<double>,
          "upper_bound"_a, DOC(sleipnir, OCPSolver, SetUpperInputBound));
  cls.def("set_upper_input_bound", &OCPSolver::SetUpperInputBound<int>,
          "upper_bound"_a, DOC(sleipnir, OCPSolver, SetUpperInputBound));
  cls.def("set_upper_input_bound", &OCPSolver::SetUpperInputBound<Variable>,
          "upper_bound"_a, DOC(sleipnir, OCPSolver, SetUpperInputBound));
  cls.def(
      "set_upper_input_bound",
      [](OCPSolver& self, nb::DRef<Eigen::MatrixXd> upperBound) {
        self.SetUpperInputBound(upperBound);
      },
      "upper_bound"_a, DOC(sleipnir, OCPSolver, SetUpperInputBound));
  cls.def("set_upper_input_bound",
          &OCPSolver::SetUpperInputBound<VariableMatrix>, "upper_bound"_a,
          DOC(sleipnir, OCPSolver, SetUpperInputBound));

  cls.def("set_min_timestep", &OCPSolver::SetMinTimestep, "min_timestep"_a,
          DOC(sleipnir, OCPSolver, SetMinTimestep));
  cls.def("set_max_timestep", &OCPSolver::SetMaxTimestep, "max_timestep"_a,
          DOC(sleipnir, OCPSolver, SetMaxTimestep));

  cls.def("X", &OCPSolver::X, DOC(sleipnir, OCPSolver, X));
  cls.def("U", &OCPSolver::U, DOC(sleipnir, OCPSolver, U));
  cls.def("DT", &OCPSolver::DT, DOC(sleipnir, OCPSolver, DT));
  cls.def("initial_state", &OCPSolver::InitialState,
          DOC(sleipnir, OCPSolver, InitialState));
  cls.def("final_state", &OCPSolver::FinalState,
          DOC(sleipnir, OCPSolver, FinalState));
}

}  // namespace sleipnir
