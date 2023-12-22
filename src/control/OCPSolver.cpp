// Copyright (c) Sleipnir contributors

#include "sleipnir/control/OCPSolver.hpp"

using namespace sleipnir;

OCPSolver::OCPSolver(int numStates, int numInputs,
                     std::chrono::duration<double> dt, int numSteps,
                     const DynamicsFunction& dynamics,
                     DynamicsType dynamicsType, TimestepMethod timestepMethod,
                     TranscriptionMethod method)
    : m_numStates{numStates},
      m_numInputs{numInputs},
      m_dt{dt},
      m_numSteps{numSteps},
      m_transcriptionMethod{method},
      m_dynamicsType{dynamicsType},
      m_dynamicsFunction{dynamics},
      m_timestepMethod{timestepMethod} {
  // u is numSteps + 1 so that the final constraintFunction evaluation works
  m_U = DecisionVariable(m_numInputs, m_numSteps + 1);

  if (m_timestepMethod == TimestepMethod::kFixed) {
    m_DT = VariableMatrix{1, m_numSteps + 1};
    for (int i = 0; i < numSteps + 1; ++i) {
      m_DT(0, i) = m_dt.count();
    }
  } else if (m_timestepMethod == TimestepMethod::kVariableSingle) {
    Variable DT = DecisionVariable();
    DT.SetValue(m_dt.count());

    // Set the member variable matrix to track the decision variable
    m_DT = VariableMatrix{1, m_numSteps + 1};
    for (int i = 0; i < numSteps + 1; ++i) {
      m_DT(0, i) = DT;
    }
  } else if (m_timestepMethod == TimestepMethod::kVariable) {
    m_DT = DecisionVariable(1, m_numSteps + 1);
    for (int i = 0; i < numSteps + 1; ++i) {
      m_DT(0, i).SetValue(m_dt.count());
    }
  }

  if (m_transcriptionMethod == TranscriptionMethod::kDirectTranscription) {
    m_X = DecisionVariable(m_numStates, m_numSteps + 1);
    ConstrainDirectTranscription();
  } else if (m_transcriptionMethod == TranscriptionMethod::kDirectCollocation) {
    m_X = DecisionVariable(m_numStates, m_numSteps + 1);
    ConstrainDirectCollocation();
  } else if (m_transcriptionMethod == TranscriptionMethod::kSingleShooting) {
    // In single-shooting the states aren't decision variables, but instead
    // depend on the input and previous states
    m_X = VariableMatrix{m_numStates, m_numSteps + 1};
    ConstrainSingleShooting();
  }
}

void OCPSolver::ConstrainDirectCollocation() {
  if (m_dynamicsType != DynamicsType::kExplicitODE) {
    throw std::runtime_error("Direct Collocation requires an explicit ODE");
  }

  Variable time = 0.0;

  for (int i = 0; i < m_numSteps; ++i) {
    auto x_begin = X().Col(i);
    auto x_end = X().Col(i + 1);
    auto u_begin = U().Col(i);
    Variable dt = DT()(0, i);
    auto t_begin = time;
    auto t_end = time + dt;
    auto t_c = t_begin + dt / 2.0;

    time += dt;

    // Use u_begin on the end point as well because we are approaching a
    // discontinuity from the left
    auto f_begin = m_dynamicsFunction(t_begin, x_begin, u_begin, dt);
    auto f_end = m_dynamicsFunction(t_end, x_end, u_begin, dt);
    auto x_c = (x_begin + x_end) / 2.0 + (f_begin - f_end) * (dt / 8.0);
    auto xprime_c =
        (x_begin - x_end) * (-3.0 / (2.0 * dt)) - (f_begin + f_end) / 4.0;
    auto f_c = m_dynamicsFunction(t_c, x_c, u_begin, dt);
    SubjectTo(f_c == xprime_c);
  }
}

void OCPSolver::ConstrainDirectTranscription() {
  Variable time = 0.0;

  for (int i = 0; i < m_numSteps; ++i) {
    auto x_begin = X().Col(i);
    auto x_end = X().Col(i + 1);
    auto u = U().Col(i);
    Variable dt = DT()(0, i);

    if (m_dynamicsType == DynamicsType::kExplicitODE) {
      SubjectTo(x_end ==
                RK4<const DynamicsFunction&, VariableMatrix, VariableMatrix,
                    Variable>(m_dynamicsFunction, x_begin, u, time, dt));
    } else if (m_dynamicsType == DynamicsType::kDiscrete) {
      SubjectTo(x_end == m_dynamicsFunction(time, x_begin, u, dt));
    }

    time += dt;
  }
}

void OCPSolver::ConstrainSingleShooting() {
  Variable time = 0.0;

  for (int i = 0; i < m_numSteps; ++i) {
    auto x_begin = X().Col(i);
    auto x_end = X().Col(i + 1);
    auto u = U().Col(i);
    Variable dt = DT()(0, i);

    if (m_dynamicsType == DynamicsType::kExplicitODE) {
      x_end = RK4<const DynamicsFunction&, VariableMatrix, VariableMatrix,
                  Variable>(m_dynamicsFunction, x_begin, u, time, dt);
    } else if (m_dynamicsType == DynamicsType::kDiscrete) {
      x_end = m_dynamicsFunction(time, x_begin, u, dt);
    }

    time += dt;
  }
}

void OCPSolver::ConstrainAlways(
    const OCPConstraintCallback& constraintFunction) {
  Variable time = 0.0;

  for (int i = 0; i < m_numSteps + 1; ++i) {
    auto x = X().Col(i);
    auto u = U().Col(i);
    auto dt = DT()(0, i);
    constraintFunction(time, x, u, dt);

    time += dt;
  }
}
