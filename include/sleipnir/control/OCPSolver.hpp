

#pragma once

#include <Eigen/Core>

#include "sleipnir/SymbolExports.hpp"
#include "sleipnir/optimization/OptimizationProblem.hpp"
#include "sleipnir/autodiff/VariableMatrix.hpp"

namespace sleipnir {

typedef std::function<VariableMatrix(double, VariableMatrix, VariableMatrix)> FixedStepODE;
typedef std::function<VariableMatrix(VariableMatrix, VariableMatrix)> FixedStepObjective;
typedef std::function<void(VariableMatrix, VariableMatrix, bool)> FixedStepConstraintFunction;

/**
 * Performs 4th order Runge-Kutta integration of dx/dt = f(t, x, u) for dt.
 *
 * @param f  The function to integrate. It must take two arguments x and u.
 * @param x  The initial value of x.
 * @param u  The value u held constant over the integration period.
 * @param t0 The initial time.
 * @param dt The time over which to integrate.
 */
template <typename T, typename U, typename V>
T RK4(std::function<T(V,T,U)> f, T x, U u, V t0, V dt) {
  auto halfdt = dt * 0.5;
  T k1 = f(t0, x, u);
  T k2 = f(t0 + halfdt, x + halfdt * k1, u);
  T k3 = f(t0 + halfdt, x + halfdt * k2, u);
  T k4 = f(t0 + dt, x + dt * k3, u);

  return x + (k1 + k2 * 2. + k3 * 2. + k4) * (dt / 6.0);
}

class SLEIPNIR_DLLEXPORT FixedStepOCPSolver {
  public:
  FixedStepOCPSolver(int N, int M, double dt, int numSteps) noexcept;
  void SetExplicitF(FixedStepODE f) { m_F = f; };
  void SetLinearDiscrete(Eigen::MatrixXd Ad, Eigen::MatrixXd Bd) { m_Ad = Ad; m_Bd = Bd; };
  void ConstrainInitialState(const VariableMatrix initialState) { m_problem.SubjectTo(X().Col(0) == initialState); };
  void ConstrainFinalState(const VariableMatrix finalState) { m_problem.SubjectTo(X().Col(m_numSteps) == finalState); };
  void ConstrainAlways(FixedStepConstraintFunction constraintFunction);
  void SetObjective(FixedStepObjective objective) { m_problem.Minimize(objective(X(), U())); };
  void SetLowerUBound(VariableMatrix lowerBound) { m_problem.SubjectTo(U() >= lowerBound); };
  void SetUpperUBound(VariableMatrix upperBound) { m_problem.SubjectTo(U() <= upperBound); };
  void DirectTranscriptionLinearDiscrete();
  void DirectTranscription();
  SolverStatus Solve();
  SolverStatus Solve(SolverConfig config);

  VariableMatrix& X() {return m_X;};
  VariableMatrix& U() {return m_U;}; 

  private:
  int m_N;
  int m_M;
  double m_dt;
  int m_numSteps;
  OptimizationProblem m_problem;
  VariableMatrix m_X;
  VariableMatrix m_U;
  FixedStepODE m_F;
  Eigen::MatrixXd m_Ad;
  Eigen::MatrixXd m_Bd;
  FixedStepObjective m_objective = [](VariableMatrix, VariableMatrix) {return VariableMatrix(0);};
  FixedStepConstraintFunction m_constraintFunction = [](VariableMatrix, VariableMatrix, bool) {};
  bool m_initialStateConstrained = false;
  bool m_finalStateConstrained = false;
  VariableMatrix m_initialState;
  VariableMatrix m_finalState;

};

}