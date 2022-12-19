

#include "sleipnir/control/OCPSolver.hpp"

using namespace sleipnir;

FixedStepOCPSolver::FixedStepOCPSolver(int N, int M, double dt, int numSteps) noexcept: m_N(N), m_M(M), m_dt(dt), m_numSteps(numSteps) {
  m_X = m_problem.DecisionVariable(m_N, m_numSteps + 1);
  m_U = m_problem.DecisionVariable(m_M, m_numSteps);
}

void FixedStepOCPSolver::DirectTranscription() {
  for (int i = 0; i < m_numSteps; ++i) {
    auto x_begin = X().Col(i);
    auto x_end = X().Col(i + 1);
    auto u = U().Col(i);
    auto t = m_dt * i;
    m_problem.SubjectTo(x_end == RK4<VariableMatrix, VariableMatrix, double>(m_F, x_begin, u, t, m_dt));
  }
}

void FixedStepOCPSolver::DirectTranscriptionLinearDiscrete() {
  for (int i = 0; i < m_numSteps; ++i) {
    auto x_begin = X().Col(i);
    auto x_end = X().Col(i + 1);
    auto u = U().Col(i);
    m_problem.SubjectTo(x_end == m_Ad * x_begin + m_Bd * u);
  }
}

void FixedStepOCPSolver::ConstrainAlways(FixedStepConstraintFunction constraintFunction) {
  // TODO: this kinda sucks and relies on the user to not add any constraints on U when i == m_numSteps
  // a simpler way is to not add the "always" constraints to the last step but that would be an even
  // worse footgun
  for (int i = 0; i < m_numSteps; ++i) {
    auto x = X().Col(i);
    auto u = U().Col(i);
    constraintFunction(x, u, false);
  }
  VariableMatrix fakeU{M, 1};
  constraintFunction(X().Col(m_numSteps, fakeU, true));
}

SolverStatus FixedStepOCPSolver::Solve() {
  return m_problem.Solve();
}

SolverStatus FixedStepOCPSolver::Solve(SolverConfig config) {
  return m_problem.Solve(config);
}