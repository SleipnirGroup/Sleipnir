// Copyright (c) Sleipnir contributors

#include <nanobind/nanobind.h>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_block.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/control/ocp.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>
#include <sleipnir/optimization/solver/iteration_info.hpp>

#include "binders.hpp"
#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

NB_MODULE(_jormungandr, m) {
  m.doc() =
      "A linearity-exploiting sparse nonlinear constrained optimization "
      "problem solver that uses the interior-point method.";

  nb::module_ autodiff = m.def_submodule("autodiff");
  nb::module_ optimization = m.def_submodule("optimization");
  nb::module_ control = m.def_submodule("control");

  nb::enum_<ExpressionType> expression_type{autodiff, "ExpressionType",
                                            DOC(slp, ExpressionType)};

  nb::class_<Variable> variable{autodiff, "Variable", DOC(slp, Variable)};
  nb::class_<VariableMatrix> variable_matrix{autodiff, "VariableMatrix",
                                             DOC(slp, VariableMatrix)};
  nb::class_<VariableBlock<VariableMatrix>> variable_block{
      autodiff, "VariableBlock", DOC(slp, VariableBlock)};

  nb::class_<Gradient> gradient{autodiff, "Gradient", DOC(slp, Gradient)};
  nb::class_<Hessian<>> hessian{autodiff, "Hessian", DOC(slp, Hessian)};
  nb::class_<Jacobian> jacobian{autodiff, "Jacobian", DOC(slp, Jacobian)};

  nb::class_<EqualityConstraints> equality_constraints{
      optimization, "EqualityConstraints", DOC(slp, EqualityConstraints)};
  nb::class_<InequalityConstraints> inequality_constraints{
      optimization, "InequalityConstraints", DOC(slp_InequalityConstraints)};

  nb::enum_<ExitStatus> exit_status{optimization, "ExitStatus",
                                    DOC(slp, ExitStatus)};
  nb::class_<IterationInfo> iteration_info{optimization, "IterationInfo",
                                           DOC(slp, IterationInfo)};

  nb::class_<Problem> problem{optimization, "Problem", DOC(slp, Problem)};

  nb::enum_<TranscriptionMethod> transcription_method{
      control, "TranscriptionMethod", DOC(slp, TranscriptionMethod)};
  nb::enum_<DynamicsType> dynamics_type{control, "DynamicsType",
                                        DOC(slp, DynamicsType)};
  nb::enum_<TimestepMethod> timestep_method{control, "TimestepMethod",
                                            DOC(slp, TimestepMethod)};
  nb::class_<OCP, Problem> ocp{control, "OCP", DOC(slp, OCP)};

  bind_expression_type(expression_type);

  bind_variable(autodiff, variable);
  bind_variable_matrix(autodiff, variable_matrix);
  bind_variable_block(variable_block);

  // Implicit conversions
  variable.def(nb::init_implicit<VariableMatrix>());
  variable_matrix.def(nb::init_implicit<VariableBlock<VariableMatrix>>());

  bind_gradient(gradient);
  bind_hessian(hessian);
  bind_jacobian(jacobian);

  bind_equality_constraints(equality_constraints);
  bind_inequality_constraints(inequality_constraints);

  bind_exit_status(exit_status);
  bind_iteration_info(iteration_info);

  bind_problem(problem);

  bind_ocp(transcription_method, dynamics_type, timestep_method, ocp);
}

}  // namespace slp
