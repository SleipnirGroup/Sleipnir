#pragma once

#include <cstdint>
#include <memory>

#include "rust/cxx.h"

namespace hafgufa_shim {
struct SolverOptions;
struct RustCallback;
struct RustDynamics;
struct RustDynamics4;
}  // namespace hafgufa_shim

namespace slp {
template <typename Scalar>
class Variable;
template <typename Scalar>
class VariableMatrix;
template <typename Scalar>
struct EqualityConstraints;
template <typename Scalar>
struct InequalityConstraints;
template <typename Scalar>
class Problem;
template <typename Scalar>
class Gradient;
// slp::Hessian is stored behind a `void*` in the shim, so we skip its
// forward declaration here — the requires-clause on its template pulls in
// Eigen headers that we don't want exposed through this shim header.
template <typename Scalar>
class Jacobian;
template <typename Scalar>
class OCP;
template <typename Scalar>
struct IterationInfo;
}  // namespace slp

namespace hafgufa_shim {

struct Variable {
  slp::Variable<double>* inner;
  Variable();
  explicit Variable(slp::Variable<double>* ptr);
  ~Variable();
  Variable(const Variable&) = delete;
  Variable& operator=(const Variable&) = delete;
};

struct VariableMatrix {
  slp::VariableMatrix<double>* inner;
  VariableMatrix();
  explicit VariableMatrix(slp::VariableMatrix<double>* ptr);
  ~VariableMatrix();
  VariableMatrix(const VariableMatrix&) = delete;
  VariableMatrix& operator=(const VariableMatrix&) = delete;
};

struct EqualityConstraints {
  slp::EqualityConstraints<double>* inner;
  EqualityConstraints();
  explicit EqualityConstraints(slp::EqualityConstraints<double>* ptr);
  ~EqualityConstraints();
  EqualityConstraints(const EqualityConstraints&) = delete;
  EqualityConstraints& operator=(const EqualityConstraints&) = delete;
};

struct InequalityConstraints {
  slp::InequalityConstraints<double>* inner;
  InequalityConstraints();
  explicit InequalityConstraints(slp::InequalityConstraints<double>* ptr);
  ~InequalityConstraints();
  InequalityConstraints(const InequalityConstraints&) = delete;
  InequalityConstraints& operator=(const InequalityConstraints&) = delete;
};

struct Problem {
  slp::Problem<double>* inner;
  Problem();
  ~Problem();
  Problem(const Problem&) = delete;
  Problem& operator=(const Problem&) = delete;
};

struct Gradient {
  slp::Gradient<double>* inner;
  Gradient();
  explicit Gradient(slp::Gradient<double>* ptr);
  ~Gradient();
  Gradient(const Gradient&) = delete;
  Gradient& operator=(const Gradient&) = delete;
};

struct Hessian {
  // slp::Hessian<double, UpLo> is templated — we box two distinct instantiations
  // behind a tag discriminator to avoid two Rust types.
  void* inner;       // slp::Hessian<double, UpLo>*
  int32_t uplo_tag;  // 0 = Lower, 1 = Lower|Upper
  Hessian();
  Hessian(void* ptr, int32_t tag);
  ~Hessian();
  Hessian(const Hessian&) = delete;
  Hessian& operator=(const Hessian&) = delete;
};

struct Jacobian {
  slp::Jacobian<double>* inner;
  Jacobian();
  explicit Jacobian(slp::Jacobian<double>* ptr);
  ~Jacobian();
  Jacobian(const Jacobian&) = delete;
  Jacobian& operator=(const Jacobian&) = delete;
};

/// Dense-triplet view of an Eigen::SparseMatrix. Row/col/value arrays are
/// parallel and have length `nnz`.
struct SparseView;

struct OCP {
  slp::OCP<double>* inner;
  OCP();
  explicit OCP(slp::OCP<double>* ptr);
  ~OCP();
  OCP(const OCP&) = delete;
  OCP& operator=(const OCP&) = delete;
};

/// Non-owning view over an `slp::IterationInfo<double>` passed by the
/// solver into a callback. Pointer is valid only for the duration of
/// the callback invocation.
struct IterationInfo {
  const slp::IterationInfo<double>* inner;
};

// Dense iterate vectors — zero-copy slices into Eigen storage.
int32_t iteration_info_iteration(const IterationInfo& info);
rust::Slice<const double> iteration_info_x(const IterationInfo& info);
rust::Slice<const double> iteration_info_s(const IterationInfo& info);
rust::Slice<const double> iteration_info_y(const IterationInfo& info);
rust::Slice<const double> iteration_info_z(const IterationInfo& info);

// Sparse cost gradient `g` — Eigen::SparseVector<double>.
int32_t iteration_info_g_size(const IterationInfo& info);
rust::Slice<const int32_t> iteration_info_g_indices(const IterationInfo& info);
rust::Slice<const double> iteration_info_g_values(const IterationInfo& info);

// Sparse Lagrangian Hessian `H` — Eigen::SparseMatrix<double> CSC form.
int32_t iteration_info_hessian_rows(const IterationInfo& info);
int32_t iteration_info_hessian_cols(const IterationInfo& info);
rust::Slice<const int32_t> iteration_info_hessian_outer(
    const IterationInfo& info);
rust::Slice<const int32_t> iteration_info_hessian_inner(
    const IterationInfo& info);
rust::Slice<const double> iteration_info_hessian_values(
    const IterationInfo& info);

// Sparse equality constraint Jacobian `A_e`.
int32_t iteration_info_eq_jacobian_rows(const IterationInfo& info);
int32_t iteration_info_eq_jacobian_cols(const IterationInfo& info);
rust::Slice<const int32_t> iteration_info_eq_jacobian_outer(
    const IterationInfo& info);
rust::Slice<const int32_t> iteration_info_eq_jacobian_inner(
    const IterationInfo& info);
rust::Slice<const double> iteration_info_eq_jacobian_values(
    const IterationInfo& info);

// Sparse inequality constraint Jacobian `A_i`.
int32_t iteration_info_ineq_jacobian_rows(const IterationInfo& info);
int32_t iteration_info_ineq_jacobian_cols(const IterationInfo& info);
rust::Slice<const int32_t> iteration_info_ineq_jacobian_outer(
    const IterationInfo& info);
rust::Slice<const int32_t> iteration_info_ineq_jacobian_inner(
    const IterationInfo& info);
rust::Slice<const double> iteration_info_ineq_jacobian_values(
    const IterationInfo& info);

// Variable lifecycle
std::unique_ptr<Variable> variable_from_f64(double value);
/// Copy a borrowed Variable into a freshly-owned FFI handle. Shares the
/// ref-counted expression node with the source.
std::unique_ptr<Variable> variable_clone_ptr(const Variable& v);
double variable_value(const Variable& v);
void variable_set_value(const Variable& v, double value);
uint8_t variable_type(const Variable& v);

// Variable arithmetic
std::unique_ptr<Variable> variable_add(const Variable& lhs, const Variable& rhs);
std::unique_ptr<Variable> variable_sub(const Variable& lhs, const Variable& rhs);
std::unique_ptr<Variable> variable_mul(const Variable& lhs, const Variable& rhs);
std::unique_ptr<Variable> variable_div(const Variable& lhs, const Variable& rhs);
std::unique_ptr<Variable> variable_neg(const Variable& v);

// Unary math
std::unique_ptr<Variable> variable_abs(const Variable& x);
std::unique_ptr<Variable> variable_acos(const Variable& x);
std::unique_ptr<Variable> variable_asin(const Variable& x);
std::unique_ptr<Variable> variable_atan(const Variable& x);
std::unique_ptr<Variable> variable_cbrt(const Variable& x);
std::unique_ptr<Variable> variable_cos(const Variable& x);
std::unique_ptr<Variable> variable_cosh(const Variable& x);
std::unique_ptr<Variable> variable_erf(const Variable& x);
std::unique_ptr<Variable> variable_exp(const Variable& x);
std::unique_ptr<Variable> variable_log(const Variable& x);
std::unique_ptr<Variable> variable_log10(const Variable& x);
std::unique_ptr<Variable> variable_sign(const Variable& x);
std::unique_ptr<Variable> variable_sin(const Variable& x);
std::unique_ptr<Variable> variable_sinh(const Variable& x);
std::unique_ptr<Variable> variable_sqrt(const Variable& x);
std::unique_ptr<Variable> variable_tan(const Variable& x);
std::unique_ptr<Variable> variable_tanh(const Variable& x);

// Binary math
std::unique_ptr<Variable> variable_atan2(const Variable& y, const Variable& x);
std::unique_ptr<Variable> variable_hypot(const Variable& x, const Variable& y);
std::unique_ptr<Variable> variable_hypot3(const Variable& x, const Variable& y,
                                          const Variable& z);
std::unique_ptr<Variable> variable_max(const Variable& a, const Variable& b);
std::unique_ptr<Variable> variable_min(const Variable& a, const Variable& b);
std::unique_ptr<Variable> variable_pow(const Variable& base,
                                       const Variable& power);

// VariableMatrix lifecycle + access
std::unique_ptr<VariableMatrix> variable_matrix_zeros(int32_t rows,
                                                     int32_t cols);
std::unique_ptr<VariableMatrix> variable_matrix_from_f64(
    int32_t rows, int32_t cols, rust::Slice<const double> data);
std::unique_ptr<VariableMatrix> variable_matrix_from_variable(const Variable& v);
std::unique_ptr<VariableMatrix> variable_matrix_clone(const VariableMatrix& m);
int32_t variable_matrix_rows(const VariableMatrix& m);
int32_t variable_matrix_cols(const VariableMatrix& m);
std::unique_ptr<Variable> variable_matrix_get(const VariableMatrix& m,
                                              int32_t row, int32_t col);
void variable_matrix_set_variable(VariableMatrix& m, int32_t row, int32_t col,
                                  const Variable& v);
void variable_matrix_set_f64(VariableMatrix& m, int32_t row, int32_t col,
                             double value);
void variable_matrix_set_value_at(VariableMatrix& m, int32_t row, int32_t col,
                                  double value);
double variable_matrix_value_at(VariableMatrix& m, int32_t row, int32_t col);
rust::Vec<double> variable_matrix_value(VariableMatrix& m);
void variable_matrix_set_value(VariableMatrix& m, rust::Slice<const double> data);
std::unique_ptr<VariableMatrix> variable_matrix_transpose(
    const VariableMatrix& m);
std::unique_ptr<VariableMatrix> variable_matrix_block(
    const VariableMatrix& m, int32_t row_offset, int32_t col_offset,
    int32_t block_rows, int32_t block_cols);
std::unique_ptr<VariableMatrix> variable_matrix_row(const VariableMatrix& m,
                                                   int32_t row);
std::unique_ptr<VariableMatrix> variable_matrix_col(const VariableMatrix& m,
                                                   int32_t col);

// VariableMatrix arithmetic
std::unique_ptr<VariableMatrix> variable_matrix_add(const VariableMatrix& lhs,
                                                    const VariableMatrix& rhs);
std::unique_ptr<VariableMatrix> variable_matrix_sub(const VariableMatrix& lhs,
                                                    const VariableMatrix& rhs);
std::unique_ptr<VariableMatrix> variable_matrix_matmul(const VariableMatrix& lhs,
                                                       const VariableMatrix& rhs);
std::unique_ptr<VariableMatrix> variable_matrix_scalar_mul(
    const VariableMatrix& lhs, const Variable& rhs);
std::unique_ptr<VariableMatrix> variable_matrix_scalar_div(
    const VariableMatrix& lhs, const Variable& rhs);
std::unique_ptr<VariableMatrix> variable_matrix_neg(const VariableMatrix& m);
std::unique_ptr<VariableMatrix> variable_matrix_hadamard(
    const VariableMatrix& lhs, const VariableMatrix& rhs);
std::unique_ptr<Variable> variable_matrix_sum(const VariableMatrix& m);

/// Solve `A * X = B` for X. Symbolic (differentiable) solve — uses
/// analytic inverses for 1x1/2x2/3x3 and a general dense solver for
/// larger systems.
std::unique_ptr<VariableMatrix> variable_matrix_solve(
    const VariableMatrix& a, const VariableMatrix& b);

/// Vertical concatenation of `top` (m₁×n) and `bottom` (m₂×n) into an
/// (m₁+m₂)×n matrix. Both operands must have the same column count.
std::unique_ptr<VariableMatrix> variable_matrix_vstack(
    const VariableMatrix& top, const VariableMatrix& bottom);
/// Horizontal concatenation of `left` (m×n₁) and `right` (m×n₂) into an
/// m×(n₁+n₂) matrix. Both operands must have the same row count.
std::unique_ptr<VariableMatrix> variable_matrix_hstack(
    const VariableMatrix& left, const VariableMatrix& right);

// Scalar constraint construction (Variable x Variable)
std::unique_ptr<EqualityConstraints> make_equality(const Variable& lhs,
                                                   const Variable& rhs);
std::unique_ptr<InequalityConstraints> make_geq(const Variable& lhs,
                                                const Variable& rhs);
std::unique_ptr<InequalityConstraints> make_leq(const Variable& lhs,
                                                const Variable& rhs);

// Matrix constraint construction (operands get broadcast when one is 1x1)
std::unique_ptr<EqualityConstraints> make_equality_matrix(
    const VariableMatrix& lhs, const VariableMatrix& rhs);
std::unique_ptr<InequalityConstraints> make_geq_matrix(
    const VariableMatrix& lhs, const VariableMatrix& rhs);
std::unique_ptr<InequalityConstraints> make_leq_matrix(
    const VariableMatrix& lhs, const VariableMatrix& rhs);

// Problem lifecycle
std::unique_ptr<Problem> problem_new();
std::unique_ptr<Variable> problem_decision_variable(Problem& problem);
std::unique_ptr<VariableMatrix> problem_decision_variable_matrix(
    Problem& problem, int32_t rows, int32_t cols);
std::unique_ptr<VariableMatrix> problem_symmetric_decision_variable(
    Problem& problem, int32_t rows);
void problem_minimize(Problem& problem, const Variable& cost);
void problem_minimize_matrix(Problem& problem, const VariableMatrix& cost);
void problem_maximize(Problem& problem, const Variable& objective);
void problem_maximize_matrix(Problem& problem, const VariableMatrix& objective);
void problem_subject_to_eq(Problem& problem, const EqualityConstraints& c);
void problem_subject_to_ineq(Problem& problem,
                             const InequalityConstraints& c);
uint8_t problem_cost_function_type(const Problem& problem);
uint8_t problem_equality_constraint_type(const Problem& problem);
uint8_t problem_inequality_constraint_type(const Problem& problem);
int8_t problem_solve(Problem& problem, SolverOptions options);

void problem_add_callback(Problem& problem,
                          rust::Box<RustCallback> callback);
void problem_add_persistent_callback(Problem& problem,
                                     rust::Box<RustCallback> callback);
void problem_clear_callbacks(Problem& problem);

// Gradient
std::unique_ptr<Gradient> gradient_new(const Variable& variable,
                                       const VariableMatrix& wrt);
rust::Vec<double> gradient_value(Gradient& gradient);
std::unique_ptr<VariableMatrix> gradient_get(const Gradient& gradient);

// Hessian. `uplo_tag`: 0 = lower-triangle only, 1 = full symmetric.
std::unique_ptr<Hessian> hessian_new(const Variable& variable,
                                     const VariableMatrix& wrt,
                                     int32_t uplo_tag);
/// Dense row-major buffer of the Hessian's sparse value, length rows*cols.
rust::Vec<double> hessian_value(Hessian& hessian);
int32_t hessian_rows(const Hessian& hessian);
int32_t hessian_cols(const Hessian& hessian);
std::unique_ptr<VariableMatrix> hessian_get(const Hessian& hessian);

// Jacobian
std::unique_ptr<Jacobian> jacobian_new(const VariableMatrix& variables,
                                       const VariableMatrix& wrt);
rust::Vec<double> jacobian_value(Jacobian& jacobian);
int32_t jacobian_rows(const Jacobian& jacobian);
int32_t jacobian_cols(const Jacobian& jacobian);
std::unique_ptr<VariableMatrix> jacobian_get(const Jacobian& jacobian);

// OCP
std::unique_ptr<OCP> ocp_new_discrete(int32_t num_states, int32_t num_inputs,
                                      double dt_seconds, int32_t num_steps,
                                      rust::Box<RustDynamics> dynamics,
                                      uint8_t timestep_method);
std::unique_ptr<OCP> ocp_new_explicit_ode(
    int32_t num_states, int32_t num_inputs, double dt_seconds,
    int32_t num_steps, rust::Box<RustDynamics> dynamics,
    uint8_t timestep_method, uint8_t transcription_method);

/// Like `ocp_new_discrete` but with a 4-arg `(t, x, u, dt)` dynamics
/// closure. Used for time-varying dynamics.
std::unique_ptr<OCP> ocp_new_discrete_4arg(int32_t num_states, int32_t num_inputs,
                                            double dt_seconds, int32_t num_steps,
                                            rust::Box<RustDynamics4> dynamics,
                                            uint8_t timestep_method);

/// Like `ocp_new_explicit_ode` but with a 4-arg `(t, x, u, dt)` dynamics
/// closure.
std::unique_ptr<OCP> ocp_new_explicit_ode_4arg(
    int32_t num_states, int32_t num_inputs, double dt_seconds,
    int32_t num_steps, rust::Box<RustDynamics4> dynamics,
    uint8_t timestep_method, uint8_t transcription_method);
void ocp_constrain_initial_state(OCP& ocp, const VariableMatrix& initial);
void ocp_constrain_final_state(OCP& ocp, const VariableMatrix& final_state);
void ocp_set_lower_input_bound_matrix(OCP& ocp, const VariableMatrix& bound);
void ocp_set_upper_input_bound_matrix(OCP& ocp, const VariableMatrix& bound);
void ocp_set_min_timestep(OCP& ocp, double min_timestep_seconds);
void ocp_set_max_timestep(OCP& ocp, double max_timestep_seconds);
std::unique_ptr<VariableMatrix> ocp_X(OCP& ocp);
std::unique_ptr<VariableMatrix> ocp_U(OCP& ocp);
std::unique_ptr<VariableMatrix> ocp_dt(OCP& ocp);
std::unique_ptr<VariableMatrix> ocp_initial_state(OCP& ocp);
std::unique_ptr<VariableMatrix> ocp_final_state(OCP& ocp);
void ocp_minimize(OCP& ocp, const Variable& cost);
void ocp_minimize_matrix(OCP& ocp, const VariableMatrix& cost);
void ocp_maximize(OCP& ocp, const Variable& objective);
void ocp_subject_to_eq(OCP& ocp, const EqualityConstraints& c);
void ocp_subject_to_ineq(OCP& ocp, const InequalityConstraints& c);
int8_t ocp_solve(OCP& ocp, SolverOptions options);
void ocp_add_callback(OCP& ocp, rust::Box<RustCallback> callback);
void ocp_add_persistent_callback(OCP& ocp,
                                 rust::Box<RustCallback> callback);
void ocp_clear_callbacks(OCP& ocp);

/// Number of control steps passed to the OCP constructor. Exposed so Rust
/// can iterate `for_each_step` externally without tracking it separately.
int32_t ocp_num_steps(const OCP& ocp);

}  // namespace hafgufa_shim
