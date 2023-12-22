// Copyright (c) Sleipnir contributors

#pragma once

#include <chrono>
#include <concepts>
#include <functional>

#include "sleipnir/autodiff/VariableMatrix.hpp"
#include "sleipnir/optimization/OptimizationProblem.hpp"
#include "sleipnir/util/Concepts.hpp"
#include "sleipnir/util/SymbolExports.hpp"

namespace sleipnir {

/**
 * Function representing an explicit or implicit ODE, or a discrete state
 * transition function.
 *
 * - Explicit: dx/dt = f(t, x, u, *)
 * - Implicit: f(t, [x dx/dt]', u, *) = 0
 * - State transition: xₖ₊₁ = f(t, xₖ, uₖ, dt)
 */
using DynamicsFunction =
    std::function<VariableMatrix(const Variable&, const VariableMatrix&,
                                 const VariableMatrix&, const Variable&)>;

/**
 * Constrain a fixed step OCP. This function is called numSteps + 1 times, once
 * for each step. The arguments are time, X_i (numStates)x1, U_i (numInputs)x1,
 * and the timestep of this segment.
 */
using OCPConstraintCallback =
    std::function<void(const Variable&, const VariableMatrix&,
                       const VariableMatrix&, const Variable&)>;

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
  State k1 = f(t0, x, u, dt);
  State k2 = f(t0 + halfdt, x + k1 * halfdt, u, dt);
  State k3 = f(t0 + halfdt, x + k2 * halfdt, u, dt);
  State k4 = f(t0 + dt, x + k3 * dt, u, dt);

  return x + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
}

/**
 * Enum describing an OCP transcription method.
 */
enum class TranscriptionMethod {
  /// Each state is a decision variable constrained to the integrated dynamics
  /// of the previous state.
  kDirectTranscription,
  /// The trajectory is modeled as a series of cubic polynomials where the
  /// centerpoint slope is constrained.
  kDirectCollocation,
  /// States depend explicitly as a function of all previous states and all
  /// previous inputs.
  kSingleShooting
};

/**
 * Enum describing a type of system dynamics constraints.
 */
enum class DynamicsType {
  /// The dynamics are a function in the form dx/dt = f(t, x, u).
  kExplicitODE,
  /// The dynamics are a function in the form xₖ₊₁ = f(t, xₖ, uₖ).
  kDiscrete
};

/**
 * Enum describing the type of system timestep.
 */
enum class TimestepMethod {
  /// The timestep is a fixed constant.
  kFixed,
  /// The timesteps are allowed to vary as independent decision variables.
  kVariable,
  /// The timesteps are equal length but allowed to vary as a single decision
  /// variable.
  kVariableSingle
};

/**
 * This class allows the user to pose and solve a constrained optimal control
 * problem (OCP) in a variety of ways.
 *
 * The system is transcripted by one of three methods (direct transcription,
 * direct collocation, or single-shooting) and additional constraints can be
 * added.
 *
 * In direct transcription, each state is a decision variable constrained to the
 * integrated dynamics of the previous state. In direct collocation, the
 * trajectory is modeled as a series of cubic polynomials where the centerpoint
 * slope is constrained. In single-shooting, states depend explicitly as a
 * function of all previous states and all previous inputs.
 *
 * Explicit ODEs are integrated using RK4.
 *
 * For explicit ODEs, the function must be in the form dx/dt = f(t, x, u).
 * For discrete state transition functions, the function must be in the form
 * xₖ₊₁ = f(t, xₖ, uₖ).
 *
 * Direct collocation requires an explicit ODE. Direct transcription and
 * single-shooting can use either an ODE or state transition function.
 *
 * https://underactuated.mit.edu/trajopt.html goes into more detail on each
 * transcription method.
 */
class SLEIPNIR_DLLEXPORT OCPSolver : public OptimizationProblem {
 public:
  /**
   * Build an optimization problem using a system evolution function (explicit
   * ODE or discrete state transition function).
   *
   * @param numStates The number of system states.
   * @param numInputs The number of system inputs.
   * @param dt The timestep for fixed-step integration.
   * @param numSteps The number of control points.
   * @param dynamics The system evolution function, either an explicit ODE or a
   * discrete state transition function.
   * @param dynamicsType The type of system evolution function.
   * @param timestepMethod The timestep method.
   * @param method The transcription method.
   */
  OCPSolver(
      int numStates, int numInputs, std::chrono::duration<double> dt,
      int numSteps, const DynamicsFunction& dynamics,
      DynamicsType dynamicsType = DynamicsType::kExplicitODE,
      TimestepMethod timestepMethod = TimestepMethod::kFixed,
      TranscriptionMethod method = TranscriptionMethod::kDirectTranscription);

  /**
   * Utility function to constrain the initial state.
   *
   * @param initialState the initial state to constrain to.
   */
  template <typename T>
    requires ScalarLike<T> || MatrixLike<T>
  void ConstrainInitialState(const T& initialState) {
    SubjectTo(InitialState() == initialState);
  }

  /**
   * Utility function to constrain the final state.
   *
   * @param finalState the final state to constrain to.
   */
  template <typename T>
    requires ScalarLike<T> || MatrixLike<T>
  void ConstrainFinalState(const T& finalState) {
    SubjectTo(FinalState() == finalState);
  }

  /**
   * Set the constraint evaluation function. This function is called
   * `numSteps+1` times, with the corresponding state and input
   * VariableMatrices.
   *
   * @param constraintFunction the constraint function.
   */
  void ConstrainAlways(const OCPConstraintCallback& constraintFunction);

  /**
   * Convenience function to set a lower bound on the input.
   *
   * @param lowerBound The lower bound that inputs must always be above. Must be
   * shaped (numInputs)x1.
   */
  template <typename T>
    requires ScalarLike<T> || MatrixLike<T>
  void SetLowerInputBound(const T& lowerBound) {
    for (int i = 0; i < m_numSteps + 1; ++i) {
      SubjectTo(U().Col(i) >= lowerBound);
    }
  }

  /**
   * Convenience function to set an upper bound on the input.
   *
   * @param upperBound The upper bound that inputs must always be below. Must be
   * shaped (numInputs)x1.
   */
  template <typename T>
    requires ScalarLike<T> || MatrixLike<T>
  void SetUpperInputBound(const T& upperBound) {
    for (int i = 0; i < m_numSteps + 1; ++i) {
      SubjectTo(U().Col(i) <= upperBound);
    }
  }

  /**
   * Convenience function to set an upper bound on the timestep.
   *
   * @param maxTimestep The maximum timestep.
   */
  void SetMaxTimestep(std::chrono::duration<double> maxTimestep) {
    SubjectTo(DT() <= maxTimestep.count());
  }

  /**
   * Convenience function to set a lower bound on the timestep.
   *
   * @param minTimestep The minimum timestep.
   */
  void SetMinTimestep(std::chrono::duration<double> minTimestep) {
    SubjectTo(DT() >= minTimestep.count());
  }

  /**
   * Get the state variables. After the problem is solved, this will contain the
   * optimized trajectory.
   *
   * Shaped (numStates)x(numSteps+1).
   *
   * @returns The state variable matrix.
   */
  VariableMatrix& X() { return m_X; };

  /**
   * Get the input variables. After the problem is solved, this will contain the
   * inputs corresponding to the optimized trajectory.
   *
   * Shaped (numInputs)x(numSteps+1), although the last input step is unused in
   * the trajectory.
   *
   * @returns The input variable matrix.
   */
  VariableMatrix& U() { return m_U; };

  /**
   * Get the timestep variables. After the problem is solved, this will contain
   * the timesteps corresponding to the optimized trajectory.
   *
   * Shaped 1x(numSteps+1), although the last timestep is unused in
   * the trajectory.
   *
   * @returns The timestep variable matrix.
   */
  VariableMatrix& DT() { return m_DT; };

  /**
   * Convenience function to get the initial state in the trajectory.
   *
   * @returns The initial state of the trajectory.
   */
  VariableMatrix InitialState() { return m_X.Col(0); }

  /**
   * Convenience function to get the final state in the trajectory.
   *
   * @returns The final state of the trajectory.
   */
  VariableMatrix FinalState() { return m_X.Col(m_numSteps); }

 private:
  void ConstrainDirectCollocation();
  void ConstrainDirectTranscription();
  void ConstrainSingleShooting();

  int m_numStates;
  int m_numInputs;
  std::chrono::duration<double> m_dt;
  int m_numSteps;
  TranscriptionMethod m_transcriptionMethod;

  DynamicsType m_dynamicsType;
  const DynamicsFunction& m_dynamicsFunction;

  TimestepMethod m_timestepMethod;

  VariableMatrix m_X;
  VariableMatrix m_U;
  VariableMatrix m_DT;
};

}  // namespace sleipnir
