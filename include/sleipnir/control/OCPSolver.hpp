

#pragma once

#include <Eigen/Core>

#include "sleipnir/SymbolExports.hpp"
#include "sleipnir/optimization/OptimizationProblem.hpp"
#include "sleipnir/autodiff/VariableMatrix.hpp"

namespace sleipnir {

/**
 * Function representing an explicit or implicit ODE, or a discrete state transition function.
 * Explicit: dx/dt = f(t, x, u) 
 * Implicit: f(t, [x dx/dt]', u) = 0
 * State transition: xₖ₊₁ = f(t, xₖ, u)
 */
using DynamicsFunction = std::function<VariableMatrix(const double&, const VariableMatrix&, const VariableMatrix&)>;
/**
 * Objective function for optimal control problems. Inputs are X := (numStates)x(numSteps + 1) and U := (numInputs)x(numSteps).
 * Return value is 1x1 cost function.
 */
using FixedStepObjective = std::function<VariableMatrix(const VariableMatrix&, const VariableMatrix&)>;
/**
 * Constrain a fixed step OCP. This function is called numSteps + 1 times, once for each step.
 * The arguments are X_i (numStates)x1, U_i (numInputs)x1, and then true if this is the last step and has no associated input, otherwise false. 
 */
using FixedStepConstraintFunction = std::function<void(const VariableMatrix&, const VariableMatrix&, const bool&)>;

/**
 * Performs 4th order Runge-Kutta integration of dx/dt = f(t, x, u) for dt.
 *
 * @param f  The function to integrate. It must take two arguments x and u.
 * @param x  The initial value of x.
 * @param u  The value u held constant over the integration period.
 * @param t0 The initial time.
 * @param dt The time over which to integrate.
 */
template <typename F, typename State, typename Input, typename Time>
State RK4(F&& f, State x, Input u, Time t0, Time dt) {
  auto halfdt = dt * 0.5;
  State k1 = f(t0, x, u);
  State k2 = f(t0 + halfdt, x + halfdt * k1, u);
  State k3 = f(t0 + halfdt, x + halfdt * k2, u);
  State k4 = f(t0 + dt, x + dt * k3, u);

  return x + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
}

enum TranscriptionMethod {
  kDirectTranscription,
  kDirectCollocation,
  kSingleShooting
};

enum DynamicsType {
  kExplicitODE,
  kImplicitODE,
  kDiscrete
};

/**
 * This class allows the user to pose and solve a constrained optimal control problem (OCP) in a variety of ways.
 */
class SLEIPNIR_DLLEXPORT FixedStepOCPSolver : public OptimizationProblem {
  public:
  /**
   * 
   * @param numStates The number of system states.
   * @param numInputs The number of system inputs.
   * @param dt The timestep for integration. Must be in seconds.
   * @param numSteps The number of control points.
   * @param dynamics The system evolution function, either an implicit or explicit ODE or a discrete state transition function.
   * @param dynamicsType The type of system evolution function.
   * @param method The transcription method.
   */
  FixedStepOCPSolver(int numStates, int numInputs, double dt, int numSteps, const DynamicsFunction& dynamics, DynamicsType dynamicsType = kExplicitODE, TranscriptionMethod method = kDirectTranscription);
  
  /**
   * Utility function to constrain the initial state.
   * 
   * @param initialState the initial state to constrain to.
   */
  void ConstrainInitialState(const VariableMatrix& initialState);

  /**
   * Utility function to constrain the final state.
   * 
   * @param initialState the final state to constrain to.
   */

  void ConstrainFinalState(const VariableMatrix& finalState);

  /**
   * Set the constraint evaluation function. This function is called `numSteps+1` times, with the corresponding state and input VariableMatrices. 
   * 
   * @param constraintFunction the constraint function.
   */
  void ConstrainAlways(const FixedStepConstraintFunction& constraintFunction);
  void SetObjective(const FixedStepObjective& objective) { Minimize(objective(X(), U())); };
  void SetLowerInputBound(const VariableMatrix& lowerBound) { SubjectTo(U() >= lowerBound); };
  void SetUpperInputBound(const VariableMatrix& upperBound) { SubjectTo(U() <= upperBound); };

  VariableMatrix& X() {return m_X;};
  VariableMatrix& U() {return m_U;};
  VariableMatrix InitialState() {return m_X.Col(0);}
  VariableMatrix FinalState() {return m_X.Col(m_numSteps + 1);}

  private:
  void ConstrainDirectCollocation();
  void ConstrainDirectTranscription();
  void ConstrainSingleShooting();

  int m_numStates;
  int m_numInputs;
  double m_dt;
  int m_numSteps;
  TranscriptionMethod m_transcriptionMethod;

  DynamicsType m_dynamicsType;
  const DynamicsFunction& m_dynamicsFunction;

  VariableMatrix m_X;
  VariableMatrix m_U;
  FixedStepConstraintFunction m_constraintFunction = [](VariableMatrix, VariableMatrix, bool) {};
};

}