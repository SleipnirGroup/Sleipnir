// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/function.h>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/control/ocp.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_ocp(nb::enum_<TranscriptionMethod>& transcription_method,
              nb::enum_<DynamicsType>& dynamics_type,
              nb::enum_<TimestepMethod>& timestep_method,
              nb::class_<OCP, Problem>& cls) {
  using namespace nb::literals;

  transcription_method.value(
      "DIRECT_TRANSCRIPTION", TranscriptionMethod::DIRECT_TRANSCRIPTION,
      DOC(slp, TranscriptionMethod, DIRECT_TRANSCRIPTION));
  transcription_method.value("DIRECT_COLLOCATION",
                             TranscriptionMethod::DIRECT_COLLOCATION,
                             DOC(slp, TranscriptionMethod, DIRECT_COLLOCATION));
  transcription_method.value("SINGLE_SHOOTING",
                             TranscriptionMethod::SINGLE_SHOOTING,
                             DOC(slp, TranscriptionMethod, SINGLE_SHOOTING));

  dynamics_type.value("EXPLICIT_ODE", DynamicsType::EXPLICIT_ODE,
                      DOC(slp, DynamicsType, EXPLICIT_ODE));
  dynamics_type.value("DISCRETE", DynamicsType::DISCRETE,
                      DOC(slp, DynamicsType, DISCRETE));

  timestep_method.value("FIXED", TimestepMethod::FIXED,
                        DOC(slp, TimestepMethod, FIXED));
  timestep_method.value("VARIABLE", TimestepMethod::VARIABLE,
                        DOC(slp, TimestepMethod, VARIABLE));
  timestep_method.value("VARIABLE_SINGLE", TimestepMethod::VARIABLE_SINGLE,
                        DOC(slp, TimestepMethod, VARIABLE_SINGLE));

  cls.def(nb::init<int, int, std::chrono::duration<double>, int,
                   const std::function<VariableMatrix(
                       const VariableMatrix& x, const VariableMatrix& u)>&,
                   DynamicsType, TimestepMethod, TranscriptionMethod>(),
          "num_states"_a, "num_inputs"_a, "dt"_a, "num_steps"_a, "dynamics"_a,
          "dynamics_type"_a = DynamicsType::EXPLICIT_ODE,
          "timestep_method"_a = TimestepMethod::FIXED,
          "transcription_method"_a = TranscriptionMethod::DIRECT_TRANSCRIPTION,
          DOC(slp, OCP, OCP));

  cls.def("constrain_initial_state", &OCP::constrain_initial_state<double>,
          "initial_state"_a, DOC(slp, OCP, constrain_initial_state));
  cls.def("constrain_initial_state", &OCP::constrain_initial_state<int>,
          "initial_state"_a, DOC(slp, OCP, constrain_initial_state));
  cls.def("constrain_initial_state", &OCP::constrain_initial_state<Variable>,
          "initial_state"_a, DOC(slp, OCP, constrain_initial_state));
  cls.def(
      "constrain_initial_state",
      [](OCP& self, nb::DRef<Eigen::MatrixXd> initial_state) {
        self.constrain_initial_state(initial_state);
      },
      "initial_state"_a, DOC(slp, OCP, constrain_initial_state));
  cls.def("constrain_initial_state",
          &OCP::constrain_initial_state<VariableMatrix>, "initial_state"_a,
          DOC(slp, OCP, constrain_initial_state));

  cls.def("constrain_final_state", &OCP::constrain_final_state<double>,
          "final_state"_a, DOC(slp, OCP, constrain_final_state));
  cls.def("constrain_final_state", &OCP::constrain_final_state<int>,
          "final_state"_a, DOC(slp, OCP, constrain_final_state));
  cls.def("constrain_final_state", &OCP::constrain_final_state<Variable>,
          "final_state"_a, DOC(slp, OCP, constrain_final_state));
  cls.def(
      "constrain_final_state",
      [](OCP& self, nb::DRef<Eigen::MatrixXd> final_state) {
        self.constrain_final_state(final_state);
      },
      "final_state"_a, DOC(slp, OCP, constrain_final_state));
  cls.def("constrain_final_state", &OCP::constrain_final_state<VariableMatrix>,
          "final_state"_a, DOC(slp, OCP, constrain_final_state));

  cls.def(
      "for_each_step",
      [](OCP& self,
         const std::function<void(const VariableMatrix& x,
                                  const VariableMatrix& u)>& callback) {
        self.for_each_step(callback);
      },
      "callback"_a, DOC(slp, OCP, for_each_step));

  cls.def("set_lower_input_bound", &OCP::set_lower_input_bound<double>,
          "lower_bound"_a, DOC(slp, OCP, set_lower_input_bound));
  cls.def("set_lower_input_bound", &OCP::set_lower_input_bound<int>,
          "lower_bound"_a, DOC(slp, OCP, set_lower_input_bound));
  cls.def("set_lower_input_bound", &OCP::set_lower_input_bound<Variable>,
          "lower_bound"_a, DOC(slp, OCP, set_lower_input_bound));
  cls.def(
      "set_lower_input_bound",
      [](OCP& self, nb::DRef<Eigen::MatrixXd> lower_bound) {
        self.set_lower_input_bound(lower_bound);
      },
      "lower_bound"_a, DOC(slp, OCP, set_lower_input_bound));
  cls.def("set_lower_input_bound", &OCP::set_lower_input_bound<VariableMatrix>,
          "lower_bound"_a, DOC(slp, OCP, set_lower_input_bound));

  cls.def("set_upper_input_bound", &OCP::set_upper_input_bound<double>,
          "upper_bound"_a, DOC(slp, OCP, set_upper_input_bound));
  cls.def("set_upper_input_bound", &OCP::set_upper_input_bound<int>,
          "upper_bound"_a, DOC(slp, OCP, set_upper_input_bound));
  cls.def("set_upper_input_bound", &OCP::set_upper_input_bound<Variable>,
          "upper_bound"_a, DOC(slp, OCP, set_upper_input_bound));
  cls.def(
      "set_upper_input_bound",
      [](OCP& self, nb::DRef<Eigen::MatrixXd> upper_bound) {
        self.set_upper_input_bound(upper_bound);
      },
      "upper_bound"_a, DOC(slp, OCP, set_upper_input_bound));
  cls.def("set_upper_input_bound", &OCP::set_upper_input_bound<VariableMatrix>,
          "upper_bound"_a, DOC(slp, OCP, set_upper_input_bound));

  cls.def("set_min_timestep", &OCP::set_min_timestep, "min_timestep"_a,
          DOC(slp, OCP, set_min_timestep));
  cls.def("set_max_timestep", &OCP::set_max_timestep, "max_timestep"_a,
          DOC(slp, OCP, set_max_timestep));

  cls.def("X", &OCP::X, DOC(slp, OCP, X));
  cls.def("U", &OCP::U, DOC(slp, OCP, U));
  cls.def("dt", &OCP::dt, DOC(slp, OCP, dt));
  cls.def("initial_state", &OCP::initial_state, DOC(slp, OCP, initial_state));
  cls.def("final_state", &OCP::final_state, DOC(slp, OCP, final_state));
}

}  // namespace slp
