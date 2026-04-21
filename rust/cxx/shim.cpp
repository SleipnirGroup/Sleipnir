#include "shim.h"

#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>

#include "hafgufa/src/ffi.rs.h"
#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/autodiff/gradient.hpp"
#include "sleipnir/autodiff/hessian.hpp"
#include "sleipnir/autodiff/jacobian.hpp"
#include "sleipnir/autodiff/variable.hpp"
#include "sleipnir/autodiff/variable_matrix.hpp"
#include "sleipnir/optimization/ocp.hpp"
#include "sleipnir/optimization/problem.hpp"
#include "sleipnir/optimization/solver/exit_status.hpp"
#include "sleipnir/optimization/solver/iteration_info.hpp"
#include "sleipnir/optimization/solver/options.hpp"

namespace hafgufa_shim {

Variable::Variable() : inner{new slp::Variable<double>()} {}
Variable::Variable(slp::Variable<double>* ptr) : inner{ptr} {}
Variable::~Variable() { delete inner; }

VariableMatrix::VariableMatrix() : inner{new slp::VariableMatrix<double>()} {}
VariableMatrix::VariableMatrix(slp::VariableMatrix<double>* ptr) : inner{ptr} {}
VariableMatrix::~VariableMatrix() { delete inner; }

EqualityConstraints::EqualityConstraints() : inner{nullptr} {}
EqualityConstraints::EqualityConstraints(slp::EqualityConstraints<double>* ptr)
    : inner{ptr} {}
EqualityConstraints::~EqualityConstraints() { delete inner; }

InequalityConstraints::InequalityConstraints() : inner{nullptr} {}
InequalityConstraints::InequalityConstraints(
    slp::InequalityConstraints<double>* ptr)
    : inner{ptr} {}
InequalityConstraints::~InequalityConstraints() { delete inner; }

Problem::Problem() : inner{new slp::Problem<double>()} {}
Problem::~Problem() { delete inner; }

Gradient::Gradient() : inner{nullptr} {}
Gradient::Gradient(slp::Gradient<double>* ptr) : inner{ptr} {}
Gradient::~Gradient() { delete inner; }

Hessian::Hessian() : inner{nullptr}, uplo_tag{1} {}
Hessian::Hessian(void* ptr, int32_t tag) : inner{ptr}, uplo_tag{tag} {}
Hessian::~Hessian() {
  if (inner == nullptr) {
    return;
  }
  if (uplo_tag == 0) {
    delete static_cast<slp::Hessian<double, Eigen::Lower>*>(inner);
  } else {
    delete static_cast<slp::Hessian<double, Eigen::Lower | Eigen::Upper>*>(inner);
  }
}

Jacobian::Jacobian() : inner{nullptr} {}
Jacobian::Jacobian(slp::Jacobian<double>* ptr) : inner{ptr} {}
Jacobian::~Jacobian() { delete inner; }

OCP::OCP() : inner{nullptr} {}
OCP::OCP(slp::OCP<double>* ptr) : inner{ptr} {}
OCP::~OCP() { delete inner; }

namespace {

std::unique_ptr<Variable> wrap(slp::Variable<double> value) {
  return std::unique_ptr<Variable>(
      new Variable{new slp::Variable<double>(std::move(value))});
}

std::unique_ptr<VariableMatrix> wrap(slp::VariableMatrix<double> value) {
  return std::unique_ptr<VariableMatrix>(new VariableMatrix{
      new slp::VariableMatrix<double>(std::move(value))});
}

std::unique_ptr<EqualityConstraints> wrap(slp::EqualityConstraints<double> c) {
  return std::unique_ptr<EqualityConstraints>(new EqualityConstraints{
      new slp::EqualityConstraints<double>(std::move(c))});
}

std::unique_ptr<InequalityConstraints> wrap(
    slp::InequalityConstraints<double> c) {
  return std::unique_ptr<InequalityConstraints>(new InequalityConstraints{
      new slp::InequalityConstraints<double>(std::move(c))});
}

}

std::unique_ptr<Variable> variable_from_f64(double value) {
  return wrap(slp::Variable<double>{value});
}

std::unique_ptr<Variable> variable_clone_ptr(const Variable& v) {
  return wrap(*v.inner);
}

double variable_value(const Variable& v) { return v.inner->value(); }

void variable_set_value(const Variable& v, double value) {
  v.inner->set_value(value);
}

uint8_t variable_type(const Variable& v) {
  return static_cast<uint8_t>(v.inner->type());
}

std::unique_ptr<Variable> variable_add(const Variable& lhs,
                                       const Variable& rhs) {
  return wrap(*lhs.inner + *rhs.inner);
}
std::unique_ptr<Variable> variable_sub(const Variable& lhs,
                                       const Variable& rhs) {
  return wrap(*lhs.inner - *rhs.inner);
}
std::unique_ptr<Variable> variable_mul(const Variable& lhs,
                                       const Variable& rhs) {
  return wrap(*lhs.inner * *rhs.inner);
}
std::unique_ptr<Variable> variable_div(const Variable& lhs,
                                       const Variable& rhs) {
  return wrap(*lhs.inner / *rhs.inner);
}
std::unique_ptr<Variable> variable_neg(const Variable& v) {
  return wrap(-*v.inner);
}

std::unique_ptr<Variable> variable_abs(const Variable& x) {
  return wrap(slp::abs(*x.inner));
}
std::unique_ptr<Variable> variable_acos(const Variable& x) {
  return wrap(slp::acos(*x.inner));
}
std::unique_ptr<Variable> variable_asin(const Variable& x) {
  return wrap(slp::asin(*x.inner));
}
std::unique_ptr<Variable> variable_atan(const Variable& x) {
  return wrap(slp::atan(*x.inner));
}
std::unique_ptr<Variable> variable_cbrt(const Variable& x) {
  return wrap(slp::cbrt(*x.inner));
}
std::unique_ptr<Variable> variable_cos(const Variable& x) {
  return wrap(slp::cos(*x.inner));
}
std::unique_ptr<Variable> variable_cosh(const Variable& x) {
  return wrap(slp::cosh(*x.inner));
}
std::unique_ptr<Variable> variable_erf(const Variable& x) {
  return wrap(slp::erf(*x.inner));
}
std::unique_ptr<Variable> variable_exp(const Variable& x) {
  return wrap(slp::exp(*x.inner));
}
std::unique_ptr<Variable> variable_log(const Variable& x) {
  return wrap(slp::log(*x.inner));
}
std::unique_ptr<Variable> variable_log10(const Variable& x) {
  return wrap(slp::log10(*x.inner));
}
std::unique_ptr<Variable> variable_sign(const Variable& x) {
  return wrap(slp::sign(*x.inner));
}
std::unique_ptr<Variable> variable_sin(const Variable& x) {
  return wrap(slp::sin(*x.inner));
}
std::unique_ptr<Variable> variable_sinh(const Variable& x) {
  return wrap(slp::sinh(*x.inner));
}
std::unique_ptr<Variable> variable_sqrt(const Variable& x) {
  return wrap(slp::sqrt(*x.inner));
}
std::unique_ptr<Variable> variable_tan(const Variable& x) {
  return wrap(slp::tan(*x.inner));
}
std::unique_ptr<Variable> variable_tanh(const Variable& x) {
  return wrap(slp::tanh(*x.inner));
}

std::unique_ptr<Variable> variable_atan2(const Variable& y, const Variable& x) {
  return wrap(slp::atan2(*y.inner, *x.inner));
}
std::unique_ptr<Variable> variable_hypot(const Variable& x, const Variable& y) {
  return wrap(slp::hypot(*x.inner, *y.inner));
}
std::unique_ptr<Variable> variable_max(const Variable& a, const Variable& b) {
  return wrap(slp::max(*a.inner, *b.inner));
}
std::unique_ptr<Variable> variable_min(const Variable& a, const Variable& b) {
  return wrap(slp::min(*a.inner, *b.inner));
}
std::unique_ptr<Variable> variable_pow(const Variable& base,
                                       const Variable& power) {
  return wrap(slp::pow(*base.inner, *power.inner));
}
std::unique_ptr<Variable> variable_hypot3(const Variable& x, const Variable& y,
                                          const Variable& z) {
  return wrap(slp::hypot(*x.inner, *y.inner, *z.inner));
}

// ---- VariableMatrix ----

std::unique_ptr<VariableMatrix> variable_matrix_zeros(int32_t rows,
                                                     int32_t cols) {
  return wrap(slp::VariableMatrix<double>(rows, cols));
}

std::unique_ptr<VariableMatrix> variable_matrix_from_f64(
    int32_t rows, int32_t cols, rust::Slice<const double> data) {
  slp::VariableMatrix<double> m(rows, cols);
  for (int32_t r = 0; r < rows; ++r) {
    for (int32_t c = 0; c < cols; ++c) {
      m[r, c] = data[r * cols + c];
    }
  }
  return wrap(std::move(m));
}

std::unique_ptr<VariableMatrix> variable_matrix_from_variable(
    const Variable& v) {
  return wrap(slp::VariableMatrix<double>(*v.inner));
}

std::unique_ptr<VariableMatrix> variable_matrix_clone(const VariableMatrix& m) {
  return wrap(*m.inner);
}

int32_t variable_matrix_rows(const VariableMatrix& m) {
  return static_cast<int32_t>(m.inner->rows());
}
int32_t variable_matrix_cols(const VariableMatrix& m) {
  return static_cast<int32_t>(m.inner->cols());
}

std::unique_ptr<Variable> variable_matrix_get(const VariableMatrix& m,
                                              int32_t row, int32_t col) {
  return wrap((*m.inner)[row, col]);
}

void variable_matrix_set_variable(VariableMatrix& m, int32_t row, int32_t col,
                                  const Variable& v) {
  (*m.inner)[row, col] = *v.inner;
}

void variable_matrix_set_f64(VariableMatrix& m, int32_t row, int32_t col,
                             double value) {
  (*m.inner)[row, col] = value;
}

void variable_matrix_set_value_at(VariableMatrix& m, int32_t row, int32_t col,
                                  double value) {
  (*m.inner)[row, col].set_value(value);
}

double variable_matrix_value_at(VariableMatrix& m, int32_t row, int32_t col) {
  return (*m.inner)[row, col].value();
}

rust::Vec<double> variable_matrix_value(VariableMatrix& m) {
  rust::Vec<double> out;
  out.reserve(static_cast<size_t>(m.inner->rows() * m.inner->cols()));
  for (int r = 0; r < m.inner->rows(); ++r) {
    for (int c = 0; c < m.inner->cols(); ++c) {
      out.push_back((*m.inner)[r, c].value());
    }
  }
  return out;
}

void variable_matrix_set_value(VariableMatrix& m, rust::Slice<const double> data) {
  int cols = m.inner->cols();
  for (int r = 0; r < m.inner->rows(); ++r) {
    for (int c = 0; c < cols; ++c) {
      (*m.inner)[r, c].set_value(data[r * cols + c]);
    }
  }
}

std::unique_ptr<VariableMatrix> variable_matrix_transpose(
    const VariableMatrix& m) {
  return wrap(m.inner->T());
}

std::unique_ptr<VariableMatrix> variable_matrix_block(
    const VariableMatrix& m, int32_t row_offset, int32_t col_offset,
    int32_t block_rows, int32_t block_cols) {
  return wrap(slp::VariableMatrix<double>(
      m.inner->block(row_offset, col_offset, block_rows, block_cols)));
}

std::unique_ptr<VariableMatrix> variable_matrix_row(const VariableMatrix& m,
                                                   int32_t row) {
  return wrap(slp::VariableMatrix<double>(m.inner->row(row)));
}

std::unique_ptr<VariableMatrix> variable_matrix_col(const VariableMatrix& m,
                                                   int32_t col) {
  return wrap(slp::VariableMatrix<double>(m.inner->col(col)));
}

std::unique_ptr<VariableMatrix> variable_matrix_add(
    const VariableMatrix& lhs, const VariableMatrix& rhs) {
  return wrap(*lhs.inner + *rhs.inner);
}
std::unique_ptr<VariableMatrix> variable_matrix_sub(
    const VariableMatrix& lhs, const VariableMatrix& rhs) {
  return wrap(*lhs.inner - *rhs.inner);
}
std::unique_ptr<VariableMatrix> variable_matrix_matmul(
    const VariableMatrix& lhs, const VariableMatrix& rhs) {
  return wrap(*lhs.inner * *rhs.inner);
}
std::unique_ptr<VariableMatrix> variable_matrix_scalar_mul(
    const VariableMatrix& lhs, const Variable& rhs) {
  return wrap(*lhs.inner * *rhs.inner);
}
std::unique_ptr<VariableMatrix> variable_matrix_scalar_div(
    const VariableMatrix& lhs, const Variable& rhs) {
  return wrap(*lhs.inner / *rhs.inner);
}
std::unique_ptr<VariableMatrix> variable_matrix_neg(const VariableMatrix& m) {
  return wrap(-*m.inner);
}

std::unique_ptr<VariableMatrix> variable_matrix_hadamard(
    const VariableMatrix& lhs, const VariableMatrix& rhs) {
  slp::VariableMatrix<double> out{slp::detail::empty, lhs.inner->rows(),
                                   lhs.inner->cols()};
  for (int r = 0; r < lhs.inner->rows(); ++r) {
    for (int c = 0; c < lhs.inner->cols(); ++c) {
      out[r, c] = (*lhs.inner)[r, c] * (*rhs.inner)[r, c];
    }
  }
  return wrap(std::move(out));
}

std::unique_ptr<Variable> variable_matrix_sum(const VariableMatrix& m) {
  slp::Variable<double> sum{0.0};
  for (int r = 0; r < m.inner->rows(); ++r) {
    for (int c = 0; c < m.inner->cols(); ++c) {
      sum += (*m.inner)[r, c];
    }
  }
  return wrap(std::move(sum));
}

std::unique_ptr<VariableMatrix> variable_matrix_solve(
    const VariableMatrix& a, const VariableMatrix& b) {
  return wrap(slp::solve(*a.inner, *b.inner));
}

std::unique_ptr<VariableMatrix> variable_matrix_vstack(
    const VariableMatrix& top, const VariableMatrix& bottom) {
  const int rows = top.inner->rows() + bottom.inner->rows();
  const int cols = top.inner->cols();
  slp::VariableMatrix<double> out{slp::detail::empty, rows, cols};
  for (int c = 0; c < cols; ++c) {
    for (int r = 0; r < top.inner->rows(); ++r) {
      out[r, c] = (*top.inner)[r, c];
    }
    for (int r = 0; r < bottom.inner->rows(); ++r) {
      out[top.inner->rows() + r, c] = (*bottom.inner)[r, c];
    }
  }
  return wrap(std::move(out));
}

std::unique_ptr<VariableMatrix> variable_matrix_hstack(
    const VariableMatrix& left, const VariableMatrix& right) {
  const int rows = left.inner->rows();
  const int cols = left.inner->cols() + right.inner->cols();
  slp::VariableMatrix<double> out{slp::detail::empty, rows, cols};
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < left.inner->cols(); ++c) {
      out[r, c] = (*left.inner)[r, c];
    }
    for (int c = 0; c < right.inner->cols(); ++c) {
      out[r, left.inner->cols() + c] = (*right.inner)[r, c];
    }
  }
  return wrap(std::move(out));
}

std::unique_ptr<EqualityConstraints> make_equality(const Variable& lhs,
                                                   const Variable& rhs) {
  return wrap(slp::EqualityConstraints<double>{*lhs.inner, *rhs.inner});
}
std::unique_ptr<InequalityConstraints> make_geq(const Variable& lhs,
                                                const Variable& rhs) {
  return wrap(slp::InequalityConstraints<double>{*lhs.inner, *rhs.inner});
}
std::unique_ptr<InequalityConstraints> make_leq(const Variable& lhs,
                                                const Variable& rhs) {
  return wrap(slp::InequalityConstraints<double>{*rhs.inner, *lhs.inner});
}

namespace {

bool is_scalar(const slp::VariableMatrix<double>& m) {
  return m.rows() == 1 && m.cols() == 1;
}

}  // namespace

std::unique_ptr<EqualityConstraints> make_equality_matrix(
    const VariableMatrix& lhs, const VariableMatrix& rhs) {
  if (is_scalar(*lhs.inner) && !is_scalar(*rhs.inner)) {
    slp::Variable<double> s = (*lhs.inner)[0, 0];
    return wrap(slp::EqualityConstraints<double>{s, *rhs.inner});
  }
  if (is_scalar(*rhs.inner) && !is_scalar(*lhs.inner)) {
    slp::Variable<double> s = (*rhs.inner)[0, 0];
    return wrap(slp::EqualityConstraints<double>{*lhs.inner, s});
  }
  return wrap(slp::EqualityConstraints<double>{*lhs.inner, *rhs.inner});
}

std::unique_ptr<InequalityConstraints> make_geq_matrix(
    const VariableMatrix& lhs, const VariableMatrix& rhs) {
  if (is_scalar(*lhs.inner) && !is_scalar(*rhs.inner)) {
    slp::Variable<double> s = (*lhs.inner)[0, 0];
    return wrap(slp::InequalityConstraints<double>{s, *rhs.inner});
  }
  if (is_scalar(*rhs.inner) && !is_scalar(*lhs.inner)) {
    slp::Variable<double> s = (*rhs.inner)[0, 0];
    return wrap(slp::InequalityConstraints<double>{*lhs.inner, s});
  }
  return wrap(slp::InequalityConstraints<double>{*lhs.inner, *rhs.inner});
}

std::unique_ptr<InequalityConstraints> make_leq_matrix(
    const VariableMatrix& lhs, const VariableMatrix& rhs) {
  return make_geq_matrix(rhs, lhs);
}

std::unique_ptr<Problem> problem_new() {
  return std::unique_ptr<Problem>(new Problem());
}

std::unique_ptr<Variable> problem_decision_variable(Problem& problem) {
  return wrap(problem.inner->decision_variable());
}

std::unique_ptr<VariableMatrix> problem_decision_variable_matrix(
    Problem& problem, int32_t rows, int32_t cols) {
  return wrap(problem.inner->decision_variable(rows, cols));
}

std::unique_ptr<VariableMatrix> problem_symmetric_decision_variable(
    Problem& problem, int32_t rows) {
  return wrap(problem.inner->symmetric_decision_variable(rows));
}

void problem_minimize(Problem& problem, const Variable& cost) {
  problem.inner->minimize(*cost.inner);
}

void problem_minimize_matrix(Problem& problem, const VariableMatrix& cost) {
  slp_assert(cost.inner->rows() == 1 && cost.inner->cols() == 1);
  problem.inner->minimize((*cost.inner)[0, 0]);
}

void problem_maximize(Problem& problem, const Variable& objective) {
  problem.inner->maximize(*objective.inner);
}

void problem_maximize_matrix(Problem& problem,
                             const VariableMatrix& objective) {
  slp_assert(objective.inner->rows() == 1 && objective.inner->cols() == 1);
  problem.inner->maximize((*objective.inner)[0, 0]);
}

void problem_subject_to_eq(Problem& problem, const EqualityConstraints& c) {
  problem.inner->subject_to(*c.inner);
}

void problem_subject_to_ineq(Problem& problem,
                             const InequalityConstraints& c) {
  problem.inner->subject_to(*c.inner);
}

uint8_t problem_cost_function_type(const Problem& problem) {
  return static_cast<uint8_t>(problem.inner->cost_function_type());
}
uint8_t problem_equality_constraint_type(const Problem& problem) {
  return static_cast<uint8_t>(problem.inner->equality_constraint_type());
}
uint8_t problem_inequality_constraint_type(const Problem& problem) {
  return static_cast<uint8_t>(problem.inner->inequality_constraint_type());
}

int8_t problem_solve(Problem& problem, SolverOptions options) {
  slp::Options opts;
  opts.tolerance = options.tolerance;
  opts.max_iterations = options.max_iterations;
  if (std::isfinite(options.timeout_seconds)) {
    opts.timeout = std::chrono::duration<double>(options.timeout_seconds);
  } else {
    opts.timeout = std::chrono::duration<double>(
        std::numeric_limits<double>::infinity());
  }
  opts.feasible_ipm = options.feasible_ipm;
  opts.diagnostics = options.diagnostics;
  return static_cast<int8_t>(problem.inner->solve(opts));
}

void problem_add_callback(Problem& problem,
                          rust::Box<RustCallback> callback) {
  auto shared = std::make_shared<rust::Box<RustCallback>>(std::move(callback));
  problem.inner->add_callback(
      [shared](const slp::IterationInfo<double>& info) -> bool {
        IterationInfo wrapper{&info};
        return (*shared)->invoke(wrapper);
      });
}

void problem_add_persistent_callback(Problem& problem,
                                     rust::Box<RustCallback> callback) {
  auto shared = std::make_shared<rust::Box<RustCallback>>(std::move(callback));
  problem.inner->add_persistent_callback(
      [shared](const slp::IterationInfo<double>& info) -> bool {
        IterationInfo wrapper{&info};
        return (*shared)->invoke(wrapper);
      });
}

void problem_clear_callbacks(Problem& problem) {
  problem.inner->clear_callbacks();
}

// ---- Gradient ----

std::unique_ptr<Gradient> gradient_new(const Variable& variable,
                                       const VariableMatrix& wrt) {
  auto* g = new slp::Gradient<double>(*variable.inner, *wrt.inner);
  return std::unique_ptr<Gradient>(new Gradient(g));
}

rust::Vec<double> gradient_value(Gradient& gradient) {
  const auto& sparse = gradient.inner->value();
  rust::Vec<double> out;
  out.reserve(static_cast<size_t>(sparse.rows()));
  for (Eigen::Index i = 0; i < sparse.rows(); ++i) {
    out.push_back(sparse.coeff(i));
  }
  return out;
}

std::unique_ptr<VariableMatrix> gradient_get(const Gradient& gradient) {
  return wrap(gradient.inner->get());
}

namespace {

template <int UpLo>
rust::Vec<double> hessian_value_impl(void* ptr) {
  auto* h = static_cast<slp::Hessian<double, UpLo>*>(ptr);
  const auto& sparse = h->value();
  rust::Vec<double> out;
  out.reserve(static_cast<size_t>(sparse.rows() * sparse.cols()));
  // Row-major dense flattening. Sparse matrix is column-major internally so
  // we explicitly transpose during iteration.
  for (Eigen::Index r = 0; r < sparse.rows(); ++r) {
    for (Eigen::Index c = 0; c < sparse.cols(); ++c) {
      out.push_back(sparse.coeff(r, c));
    }
  }
  return out;
}

}

std::unique_ptr<Hessian> hessian_new(const Variable& variable,
                                     const VariableMatrix& wrt,
                                     int32_t uplo_tag) {
  if (uplo_tag == 0) {
    auto* h = new slp::Hessian<double, Eigen::Lower>(*variable.inner,
                                                      *wrt.inner);
    return std::unique_ptr<Hessian>(new Hessian(h, 0));
  } else {
    auto* h = new slp::Hessian<double, Eigen::Lower | Eigen::Upper>(
        *variable.inner, *wrt.inner);
    return std::unique_ptr<Hessian>(new Hessian(h, 1));
  }
}

rust::Vec<double> hessian_value(Hessian& hessian) {
  if (hessian.uplo_tag == 0) {
    return hessian_value_impl<Eigen::Lower>(hessian.inner);
  } else {
    return hessian_value_impl<Eigen::Lower | Eigen::Upper>(hessian.inner);
  }
}

int32_t hessian_rows(const Hessian& hessian) {
  if (hessian.uplo_tag == 0) {
    return static_cast<int32_t>(
        static_cast<slp::Hessian<double, Eigen::Lower>*>(hessian.inner)
            ->value()
            .rows());
  } else {
    return static_cast<int32_t>(
        static_cast<slp::Hessian<double, Eigen::Lower | Eigen::Upper>*>(
            hessian.inner)
            ->value()
            .rows());
  }
}

int32_t hessian_cols(const Hessian& hessian) {
  if (hessian.uplo_tag == 0) {
    return static_cast<int32_t>(
        static_cast<slp::Hessian<double, Eigen::Lower>*>(hessian.inner)
            ->value()
            .cols());
  } else {
    return static_cast<int32_t>(
        static_cast<slp::Hessian<double, Eigen::Lower | Eigen::Upper>*>(
            hessian.inner)
            ->value()
            .cols());
  }
}

std::unique_ptr<VariableMatrix> hessian_get(const Hessian& hessian) {
  if (hessian.uplo_tag == 0) {
    auto* h = static_cast<slp::Hessian<double, Eigen::Lower>*>(hessian.inner);
    return wrap(h->get());
  } else {
    auto* h = static_cast<slp::Hessian<double, Eigen::Lower | Eigen::Upper>*>(
        hessian.inner);
    return wrap(h->get());
  }
}

std::unique_ptr<Jacobian> jacobian_new(const VariableMatrix& variables,
                                       const VariableMatrix& wrt) {
  auto* j = new slp::Jacobian<double>(*variables.inner, *wrt.inner);
  return std::unique_ptr<Jacobian>(new Jacobian(j));
}

rust::Vec<double> jacobian_value(Jacobian& jacobian) {
  const auto& sparse = jacobian.inner->value();
  rust::Vec<double> out;
  out.reserve(static_cast<size_t>(sparse.rows() * sparse.cols()));
  for (Eigen::Index r = 0; r < sparse.rows(); ++r) {
    for (Eigen::Index c = 0; c < sparse.cols(); ++c) {
      out.push_back(sparse.coeff(r, c));
    }
  }
  return out;
}

int32_t jacobian_rows(const Jacobian& jacobian) {
  return static_cast<int32_t>(jacobian.inner->value().rows());
}

int32_t jacobian_cols(const Jacobian& jacobian) {
  return static_cast<int32_t>(jacobian.inner->value().cols());
}

std::unique_ptr<VariableMatrix> jacobian_get(const Jacobian& jacobian) {
  return wrap(jacobian.inner->get());
}

namespace {

slp::VariableMatrix<double> invoke_rust_dynamics(
    rust::Box<RustDynamics>& dyn, const slp::VariableMatrix<double>& x,
    const slp::VariableMatrix<double>& u) {
  VariableMatrix x_wrap(const_cast<slp::VariableMatrix<double>*>(&x));
  VariableMatrix u_wrap(const_cast<slp::VariableMatrix<double>*>(&u));
  struct Guard {
    VariableMatrix* wrapper;
    ~Guard() { wrapper->inner = nullptr; }
  };
  Guard gx{&x_wrap};
  Guard gu{&u_wrap};
  auto result = dyn->invoke(x_wrap, u_wrap);
  slp::VariableMatrix<double> out = *result->inner;
  return out;
}

}

std::unique_ptr<OCP> ocp_new_discrete(int32_t num_states, int32_t num_inputs,
                                      double dt_seconds, int32_t num_steps,
                                      rust::Box<RustDynamics> dynamics,
                                      uint8_t timestep_method) {
  auto shared =
      std::make_shared<rust::Box<RustDynamics>>(std::move(dynamics));
  auto dyn_fn = [shared](const slp::VariableMatrix<double>& x,
                         const slp::VariableMatrix<double>& u)
      -> slp::VariableMatrix<double> {
    return invoke_rust_dynamics(*shared, x, u);
  };
  auto* ocp = new slp::OCP<double>(
      num_states, num_inputs, std::chrono::duration<double>(dt_seconds),
      num_steps, dyn_fn, slp::DynamicsType::DISCRETE,
      static_cast<slp::TimestepMethod>(timestep_method),
      slp::TranscriptionMethod::DIRECT_TRANSCRIPTION);
  return std::unique_ptr<OCP>(new OCP(ocp));
}

std::unique_ptr<OCP> ocp_new_explicit_ode(
    int32_t num_states, int32_t num_inputs, double dt_seconds,
    int32_t num_steps, rust::Box<RustDynamics> dynamics,
    uint8_t timestep_method, uint8_t transcription_method) {
  auto shared =
      std::make_shared<rust::Box<RustDynamics>>(std::move(dynamics));
  auto dyn_fn = [shared](const slp::VariableMatrix<double>& x,
                         const slp::VariableMatrix<double>& u)
      -> slp::VariableMatrix<double> {
    return invoke_rust_dynamics(*shared, x, u);
  };
  auto* ocp = new slp::OCP<double>(
      num_states, num_inputs, std::chrono::duration<double>(dt_seconds),
      num_steps, dyn_fn, slp::DynamicsType::EXPLICIT_ODE,
      static_cast<slp::TimestepMethod>(timestep_method),
      static_cast<slp::TranscriptionMethod>(transcription_method));
  return std::unique_ptr<OCP>(new OCP(ocp));
}

namespace {

slp::VariableMatrix<double> invoke_rust_dynamics4(
    rust::Box<RustDynamics4>& dyn, const slp::Variable<double>& t,
    const slp::VariableMatrix<double>& x,
    const slp::VariableMatrix<double>& u, const slp::Variable<double>& dt) {
  // Wrap the four operands as non-owning shim handles; the Rust closure
  // clones them into arena-owned copies before the wrappers expire.
  Variable t_wrap{const_cast<slp::Variable<double>*>(&t)};
  VariableMatrix x_wrap(const_cast<slp::VariableMatrix<double>*>(&x));
  VariableMatrix u_wrap(const_cast<slp::VariableMatrix<double>*>(&u));
  Variable dt_wrap{const_cast<slp::Variable<double>*>(&dt)};
  struct Guard {
    Variable* v;
    ~Guard() { v->inner = nullptr; }
  };
  struct MGuard {
    VariableMatrix* m;
    ~MGuard() { m->inner = nullptr; }
  };
  Guard gt{&t_wrap};
  MGuard gx{&x_wrap};
  MGuard gu{&u_wrap};
  Guard gdt{&dt_wrap};
  auto result = dyn->invoke(t_wrap, x_wrap, u_wrap, dt_wrap);
  slp::VariableMatrix<double> out = *result->inner;
  return out;
}

}  // namespace

std::unique_ptr<OCP> ocp_new_discrete_4arg(int32_t num_states,
                                            int32_t num_inputs,
                                            double dt_seconds,
                                            int32_t num_steps,
                                            rust::Box<RustDynamics4> dynamics,
                                            uint8_t timestep_method) {
  auto shared =
      std::make_shared<rust::Box<RustDynamics4>>(std::move(dynamics));
  auto dyn_fn = [shared](const slp::Variable<double>& t,
                         const slp::VariableMatrix<double>& x,
                         const slp::VariableMatrix<double>& u,
                         const slp::Variable<double>& dt)
      -> slp::VariableMatrix<double> {
    return invoke_rust_dynamics4(*shared, t, x, u, dt);
  };
  auto* ocp = new slp::OCP<double>(
      num_states, num_inputs, std::chrono::duration<double>(dt_seconds),
      num_steps, dyn_fn, slp::DynamicsType::DISCRETE,
      static_cast<slp::TimestepMethod>(timestep_method),
      slp::TranscriptionMethod::DIRECT_TRANSCRIPTION);
  return std::unique_ptr<OCP>(new OCP(ocp));
}

std::unique_ptr<OCP> ocp_new_explicit_ode_4arg(
    int32_t num_states, int32_t num_inputs, double dt_seconds,
    int32_t num_steps, rust::Box<RustDynamics4> dynamics,
    uint8_t timestep_method, uint8_t transcription_method) {
  auto shared =
      std::make_shared<rust::Box<RustDynamics4>>(std::move(dynamics));
  auto dyn_fn = [shared](const slp::Variable<double>& t,
                         const slp::VariableMatrix<double>& x,
                         const slp::VariableMatrix<double>& u,
                         const slp::Variable<double>& dt)
      -> slp::VariableMatrix<double> {
    return invoke_rust_dynamics4(*shared, t, x, u, dt);
  };
  auto* ocp = new slp::OCP<double>(
      num_states, num_inputs, std::chrono::duration<double>(dt_seconds),
      num_steps, dyn_fn, slp::DynamicsType::EXPLICIT_ODE,
      static_cast<slp::TimestepMethod>(timestep_method),
      static_cast<slp::TranscriptionMethod>(transcription_method));
  return std::unique_ptr<OCP>(new OCP(ocp));
}

void ocp_constrain_initial_state(OCP& ocp, const VariableMatrix& initial) {
  ocp.inner->constrain_initial_state(*initial.inner);
}

void ocp_constrain_final_state(OCP& ocp, const VariableMatrix& final_state) {
  ocp.inner->constrain_final_state(*final_state.inner);
}

void ocp_set_lower_input_bound_matrix(OCP& ocp, const VariableMatrix& bound) {
  ocp.inner->set_lower_input_bound(*bound.inner);
}

void ocp_set_upper_input_bound_matrix(OCP& ocp, const VariableMatrix& bound) {
  ocp.inner->set_upper_input_bound(*bound.inner);
}

void ocp_set_min_timestep(OCP& ocp, double min_timestep_seconds) {
  ocp.inner->set_min_timestep(
      std::chrono::duration<double>(min_timestep_seconds));
}

void ocp_set_max_timestep(OCP& ocp, double max_timestep_seconds) {
  ocp.inner->set_max_timestep(
      std::chrono::duration<double>(max_timestep_seconds));
}

std::unique_ptr<VariableMatrix> ocp_X(OCP& ocp) { return wrap(ocp.inner->X()); }
std::unique_ptr<VariableMatrix> ocp_U(OCP& ocp) { return wrap(ocp.inner->U()); }
std::unique_ptr<VariableMatrix> ocp_dt(OCP& ocp) {
  return wrap(ocp.inner->dt());
}
std::unique_ptr<VariableMatrix> ocp_initial_state(OCP& ocp) {
  return wrap(ocp.inner->initial_state());
}
std::unique_ptr<VariableMatrix> ocp_final_state(OCP& ocp) {
  return wrap(ocp.inner->final_state());
}

void ocp_minimize(OCP& ocp, const Variable& cost) {
  ocp.inner->minimize(*cost.inner);
}

void ocp_minimize_matrix(OCP& ocp, const VariableMatrix& cost) {
  slp_assert(cost.inner->rows() == 1 && cost.inner->cols() == 1);
  ocp.inner->minimize((*cost.inner)[0, 0]);
}

void ocp_maximize(OCP& ocp, const Variable& objective) {
  ocp.inner->maximize(*objective.inner);
}

void ocp_subject_to_eq(OCP& ocp, const EqualityConstraints& c) {
  ocp.inner->subject_to(*c.inner);
}

void ocp_subject_to_ineq(OCP& ocp, const InequalityConstraints& c) {
  ocp.inner->subject_to(*c.inner);
}

int8_t ocp_solve(OCP& ocp, SolverOptions options) {
  slp::Options opts;
  opts.tolerance = options.tolerance;
  opts.max_iterations = options.max_iterations;
  if (std::isfinite(options.timeout_seconds)) {
    opts.timeout = std::chrono::duration<double>(options.timeout_seconds);
  } else {
    opts.timeout = std::chrono::duration<double>(
        std::numeric_limits<double>::infinity());
  }
  opts.feasible_ipm = options.feasible_ipm;
  opts.diagnostics = options.diagnostics;
  return static_cast<int8_t>(ocp.inner->solve(opts));
}

void ocp_add_callback(OCP& ocp, rust::Box<RustCallback> callback) {
  auto shared = std::make_shared<rust::Box<RustCallback>>(std::move(callback));
  ocp.inner->add_callback(
      [shared](const slp::IterationInfo<double>& info) -> bool {
        IterationInfo wrapper{&info};
        return (*shared)->invoke(wrapper);
      });
}

void ocp_clear_callbacks(OCP& ocp) { ocp.inner->clear_callbacks(); }

void ocp_add_persistent_callback(OCP& ocp,
                                 rust::Box<RustCallback> callback) {
  auto shared = std::make_shared<rust::Box<RustCallback>>(std::move(callback));
  ocp.inner->add_persistent_callback(
      [shared](const slp::IterationInfo<double>& info) -> bool {
        IterationInfo wrapper{&info};
        return (*shared)->invoke(wrapper);
      });
}

int32_t ocp_num_steps(const OCP& ocp) {
  return static_cast<int32_t>(ocp.inner->U().cols() - 1);
}

// ---- IterationInfo accessors ----

namespace {
template <typename Vec>
rust::Slice<const double> dense_slice(const Vec& v) {
  return rust::Slice<const double>{v.data(), static_cast<size_t>(v.size())};
}
}  // namespace

int32_t iteration_info_iteration(const IterationInfo& info) {
  return info.inner->iteration;
}

rust::Slice<const double> iteration_info_x(const IterationInfo& info) {
  return dense_slice(info.inner->x);
}
rust::Slice<const double> iteration_info_s(const IterationInfo& info) {
  return dense_slice(info.inner->s);
}
rust::Slice<const double> iteration_info_y(const IterationInfo& info) {
  return dense_slice(info.inner->y);
}
rust::Slice<const double> iteration_info_z(const IterationInfo& info) {
  return dense_slice(info.inner->z);
}

int32_t iteration_info_g_size(const IterationInfo& info) {
  return static_cast<int32_t>(info.inner->g.size());
}
rust::Slice<const int32_t> iteration_info_g_indices(
    const IterationInfo& info) {
  return rust::Slice<const int32_t>{
      info.inner->g.innerIndexPtr(),
      static_cast<size_t>(info.inner->g.nonZeros())};
}
rust::Slice<const double> iteration_info_g_values(const IterationInfo& info) {
  return rust::Slice<const double>{
      info.inner->g.valuePtr(),
      static_cast<size_t>(info.inner->g.nonZeros())};
}

// Sparse matrices store CSC: outer index length = cols + 1, inner/values
// length = nnz.

int32_t iteration_info_hessian_rows(const IterationInfo& info) {
  return static_cast<int32_t>(info.inner->H.rows());
}
int32_t iteration_info_hessian_cols(const IterationInfo& info) {
  return static_cast<int32_t>(info.inner->H.cols());
}
rust::Slice<const int32_t> iteration_info_hessian_outer(
    const IterationInfo& info) {
  return rust::Slice<const int32_t>{
      info.inner->H.outerIndexPtr(),
      static_cast<size_t>(info.inner->H.outerSize() + 1)};
}
rust::Slice<const int32_t> iteration_info_hessian_inner(
    const IterationInfo& info) {
  return rust::Slice<const int32_t>{
      info.inner->H.innerIndexPtr(),
      static_cast<size_t>(info.inner->H.nonZeros())};
}
rust::Slice<const double> iteration_info_hessian_values(
    const IterationInfo& info) {
  return rust::Slice<const double>{
      info.inner->H.valuePtr(),
      static_cast<size_t>(info.inner->H.nonZeros())};
}

int32_t iteration_info_eq_jacobian_rows(const IterationInfo& info) {
  return static_cast<int32_t>(info.inner->A_e.rows());
}
int32_t iteration_info_eq_jacobian_cols(const IterationInfo& info) {
  return static_cast<int32_t>(info.inner->A_e.cols());
}
rust::Slice<const int32_t> iteration_info_eq_jacobian_outer(
    const IterationInfo& info) {
  return rust::Slice<const int32_t>{
      info.inner->A_e.outerIndexPtr(),
      static_cast<size_t>(info.inner->A_e.outerSize() + 1)};
}
rust::Slice<const int32_t> iteration_info_eq_jacobian_inner(
    const IterationInfo& info) {
  return rust::Slice<const int32_t>{
      info.inner->A_e.innerIndexPtr(),
      static_cast<size_t>(info.inner->A_e.nonZeros())};
}
rust::Slice<const double> iteration_info_eq_jacobian_values(
    const IterationInfo& info) {
  return rust::Slice<const double>{
      info.inner->A_e.valuePtr(),
      static_cast<size_t>(info.inner->A_e.nonZeros())};
}

int32_t iteration_info_ineq_jacobian_rows(const IterationInfo& info) {
  return static_cast<int32_t>(info.inner->A_i.rows());
}
int32_t iteration_info_ineq_jacobian_cols(const IterationInfo& info) {
  return static_cast<int32_t>(info.inner->A_i.cols());
}
rust::Slice<const int32_t> iteration_info_ineq_jacobian_outer(
    const IterationInfo& info) {
  return rust::Slice<const int32_t>{
      info.inner->A_i.outerIndexPtr(),
      static_cast<size_t>(info.inner->A_i.outerSize() + 1)};
}
rust::Slice<const int32_t> iteration_info_ineq_jacobian_inner(
    const IterationInfo& info) {
  return rust::Slice<const int32_t>{
      info.inner->A_i.innerIndexPtr(),
      static_cast<size_t>(info.inner->A_i.nonZeros())};
}
rust::Slice<const double> iteration_info_ineq_jacobian_values(
    const IterationInfo& info) {
  return rust::Slice<const double>{
      info.inner->A_i.valuePtr(),
      static_cast<size_t>(info.inner->A_i.nonZeros())};
}

}  // namespace hafgufa_shim
