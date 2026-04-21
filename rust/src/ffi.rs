#[cxx::bridge(namespace = "hafgufa_shim")]
pub(crate) mod ffi {
    /// Plain-old-data mirror of `slp::Options`, marshalled across FFI.
    #[derive(Clone, Copy, Debug)]
    pub struct SolverOptions {
        pub tolerance: f64,
        pub max_iterations: i32,
        pub timeout_seconds: f64,
        pub feasible_ipm: bool,
        pub diagnostics: bool,
    }

    extern "Rust" {
        type RustCallback;
        fn invoke(self: &mut RustCallback, info: &IterationInfo) -> bool;

        type RustDynamics;
        fn invoke(
            self: &mut RustDynamics,
            x: &VariableMatrix,
            u: &VariableMatrix,
        ) -> UniquePtr<VariableMatrix>;

        type RustDynamics4;
        fn invoke(
            self: &mut RustDynamics4,
            t: &Variable,
            x: &VariableMatrix,
            u: &VariableMatrix,
            dt: &Variable,
        ) -> UniquePtr<VariableMatrix>;
    }

    unsafe extern "C++" {
        include!("shim.h");

        type Variable;
        type VariableMatrix;
        type EqualityConstraints;
        type InequalityConstraints;
        type Problem;
        type Gradient;
        type Hessian;
        type Jacobian;
        type OCP;
        type IterationInfo;

        // IterationInfo — zero-copy slices into Eigen storage; valid
        // only for the duration of the callback invocation.
        fn iteration_info_iteration(info: &IterationInfo) -> i32;
        fn iteration_info_x(info: &IterationInfo) -> &[f64];
        fn iteration_info_s(info: &IterationInfo) -> &[f64];
        fn iteration_info_y(info: &IterationInfo) -> &[f64];
        fn iteration_info_z(info: &IterationInfo) -> &[f64];

        fn iteration_info_g_size(info: &IterationInfo) -> i32;
        fn iteration_info_g_indices(info: &IterationInfo) -> &[i32];
        fn iteration_info_g_values(info: &IterationInfo) -> &[f64];

        fn iteration_info_hessian_rows(info: &IterationInfo) -> i32;
        fn iteration_info_hessian_cols(info: &IterationInfo) -> i32;
        fn iteration_info_hessian_outer(info: &IterationInfo) -> &[i32];
        fn iteration_info_hessian_inner(info: &IterationInfo) -> &[i32];
        fn iteration_info_hessian_values(info: &IterationInfo) -> &[f64];

        fn iteration_info_eq_jacobian_rows(info: &IterationInfo) -> i32;
        fn iteration_info_eq_jacobian_cols(info: &IterationInfo) -> i32;
        fn iteration_info_eq_jacobian_outer(info: &IterationInfo) -> &[i32];
        fn iteration_info_eq_jacobian_inner(info: &IterationInfo) -> &[i32];
        fn iteration_info_eq_jacobian_values(info: &IterationInfo) -> &[f64];

        fn iteration_info_ineq_jacobian_rows(info: &IterationInfo) -> i32;
        fn iteration_info_ineq_jacobian_cols(info: &IterationInfo) -> i32;
        fn iteration_info_ineq_jacobian_outer(info: &IterationInfo) -> &[i32];
        fn iteration_info_ineq_jacobian_inner(info: &IterationInfo) -> &[i32];
        fn iteration_info_ineq_jacobian_values(info: &IterationInfo) -> &[f64];

        // Variable
        fn variable_from_f64(value: f64) -> UniquePtr<Variable>;
        fn variable_clone_ptr(v: &Variable) -> UniquePtr<Variable>;
        fn variable_value(v: &Variable) -> f64;
        fn variable_set_value(v: &Variable, value: f64);
        fn variable_type(v: &Variable) -> u8;

        fn variable_add(lhs: &Variable, rhs: &Variable) -> UniquePtr<Variable>;
        fn variable_sub(lhs: &Variable, rhs: &Variable) -> UniquePtr<Variable>;
        fn variable_mul(lhs: &Variable, rhs: &Variable) -> UniquePtr<Variable>;
        fn variable_div(lhs: &Variable, rhs: &Variable) -> UniquePtr<Variable>;
        fn variable_neg(v: &Variable) -> UniquePtr<Variable>;

        fn variable_abs(x: &Variable) -> UniquePtr<Variable>;
        fn variable_acos(x: &Variable) -> UniquePtr<Variable>;
        fn variable_asin(x: &Variable) -> UniquePtr<Variable>;
        fn variable_atan(x: &Variable) -> UniquePtr<Variable>;
        fn variable_cbrt(x: &Variable) -> UniquePtr<Variable>;
        fn variable_cos(x: &Variable) -> UniquePtr<Variable>;
        fn variable_cosh(x: &Variable) -> UniquePtr<Variable>;
        fn variable_erf(x: &Variable) -> UniquePtr<Variable>;
        fn variable_exp(x: &Variable) -> UniquePtr<Variable>;
        fn variable_log(x: &Variable) -> UniquePtr<Variable>;
        fn variable_log10(x: &Variable) -> UniquePtr<Variable>;
        fn variable_sign(x: &Variable) -> UniquePtr<Variable>;
        fn variable_sin(x: &Variable) -> UniquePtr<Variable>;
        fn variable_sinh(x: &Variable) -> UniquePtr<Variable>;
        fn variable_sqrt(x: &Variable) -> UniquePtr<Variable>;
        fn variable_tan(x: &Variable) -> UniquePtr<Variable>;
        fn variable_tanh(x: &Variable) -> UniquePtr<Variable>;

        fn variable_atan2(y: &Variable, x: &Variable) -> UniquePtr<Variable>;
        fn variable_hypot(x: &Variable, y: &Variable) -> UniquePtr<Variable>;
        fn variable_hypot3(
            x: &Variable,
            y: &Variable,
            z: &Variable,
        ) -> UniquePtr<Variable>;
        fn variable_max(a: &Variable, b: &Variable) -> UniquePtr<Variable>;
        fn variable_min(a: &Variable, b: &Variable) -> UniquePtr<Variable>;
        fn variable_pow(base: &Variable, power: &Variable) -> UniquePtr<Variable>;

        // VariableMatrix
        fn variable_matrix_zeros(rows: i32, cols: i32) -> UniquePtr<VariableMatrix>;
        fn variable_matrix_from_f64(
            rows: i32,
            cols: i32,
            data: &[f64],
        ) -> UniquePtr<VariableMatrix>;
        fn variable_matrix_from_variable(v: &Variable) -> UniquePtr<VariableMatrix>;
        fn variable_matrix_clone(m: &VariableMatrix) -> UniquePtr<VariableMatrix>;
        fn variable_matrix_rows(m: &VariableMatrix) -> i32;
        fn variable_matrix_cols(m: &VariableMatrix) -> i32;
        fn variable_matrix_get(
            m: &VariableMatrix,
            row: i32,
            col: i32,
        ) -> UniquePtr<Variable>;
        fn variable_matrix_set_variable(
            m: Pin<&mut VariableMatrix>,
            row: i32,
            col: i32,
            v: &Variable,
        );
        fn variable_matrix_set_f64(
            m: Pin<&mut VariableMatrix>,
            row: i32,
            col: i32,
            value: f64,
        );
        fn variable_matrix_set_value_at(
            m: Pin<&mut VariableMatrix>,
            row: i32,
            col: i32,
            value: f64,
        );
        fn variable_matrix_value_at(
            m: Pin<&mut VariableMatrix>,
            row: i32,
            col: i32,
        ) -> f64;
        fn variable_matrix_value(m: Pin<&mut VariableMatrix>) -> Vec<f64>;
        fn variable_matrix_set_value(m: Pin<&mut VariableMatrix>, data: &[f64]);
        fn variable_matrix_transpose(m: &VariableMatrix) -> UniquePtr<VariableMatrix>;
        fn variable_matrix_block(
            m: &VariableMatrix,
            row_offset: i32,
            col_offset: i32,
            block_rows: i32,
            block_cols: i32,
        ) -> UniquePtr<VariableMatrix>;
        fn variable_matrix_row(m: &VariableMatrix, row: i32) -> UniquePtr<VariableMatrix>;
        fn variable_matrix_col(m: &VariableMatrix, col: i32) -> UniquePtr<VariableMatrix>;

        fn variable_matrix_add(
            lhs: &VariableMatrix,
            rhs: &VariableMatrix,
        ) -> UniquePtr<VariableMatrix>;
        fn variable_matrix_sub(
            lhs: &VariableMatrix,
            rhs: &VariableMatrix,
        ) -> UniquePtr<VariableMatrix>;
        fn variable_matrix_matmul(
            lhs: &VariableMatrix,
            rhs: &VariableMatrix,
        ) -> UniquePtr<VariableMatrix>;
        fn variable_matrix_scalar_mul(
            lhs: &VariableMatrix,
            rhs: &Variable,
        ) -> UniquePtr<VariableMatrix>;
        fn variable_matrix_scalar_div(
            lhs: &VariableMatrix,
            rhs: &Variable,
        ) -> UniquePtr<VariableMatrix>;
        fn variable_matrix_neg(m: &VariableMatrix) -> UniquePtr<VariableMatrix>;
        fn variable_matrix_hadamard(
            lhs: &VariableMatrix,
            rhs: &VariableMatrix,
        ) -> UniquePtr<VariableMatrix>;
        fn variable_matrix_sum(m: &VariableMatrix) -> UniquePtr<Variable>;
        fn variable_matrix_solve(
            a: &VariableMatrix,
            b: &VariableMatrix,
        ) -> UniquePtr<VariableMatrix>;
        fn variable_matrix_vstack(
            top: &VariableMatrix,
            bottom: &VariableMatrix,
        ) -> UniquePtr<VariableMatrix>;
        fn variable_matrix_hstack(
            left: &VariableMatrix,
            right: &VariableMatrix,
        ) -> UniquePtr<VariableMatrix>;

        // Scalar constraints
        fn make_equality(lhs: &Variable, rhs: &Variable) -> UniquePtr<EqualityConstraints>;
        fn make_geq(lhs: &Variable, rhs: &Variable) -> UniquePtr<InequalityConstraints>;
        fn make_leq(lhs: &Variable, rhs: &Variable) -> UniquePtr<InequalityConstraints>;

        // Matrix constraints
        fn make_equality_matrix(
            lhs: &VariableMatrix,
            rhs: &VariableMatrix,
        ) -> UniquePtr<EqualityConstraints>;
        fn make_geq_matrix(
            lhs: &VariableMatrix,
            rhs: &VariableMatrix,
        ) -> UniquePtr<InequalityConstraints>;
        fn make_leq_matrix(
            lhs: &VariableMatrix,
            rhs: &VariableMatrix,
        ) -> UniquePtr<InequalityConstraints>;

        // Problem
        fn problem_new() -> UniquePtr<Problem>;
        fn problem_decision_variable(problem: Pin<&mut Problem>) -> UniquePtr<Variable>;
        fn problem_decision_variable_matrix(
            problem: Pin<&mut Problem>,
            rows: i32,
            cols: i32,
        ) -> UniquePtr<VariableMatrix>;
        fn problem_symmetric_decision_variable(
            problem: Pin<&mut Problem>,
            rows: i32,
        ) -> UniquePtr<VariableMatrix>;
        fn problem_minimize(problem: Pin<&mut Problem>, cost: &Variable);
        fn problem_minimize_matrix(problem: Pin<&mut Problem>, cost: &VariableMatrix);
        fn problem_maximize(problem: Pin<&mut Problem>, objective: &Variable);
        fn problem_maximize_matrix(
            problem: Pin<&mut Problem>,
            objective: &VariableMatrix,
        );
        fn problem_subject_to_eq(problem: Pin<&mut Problem>, c: &EqualityConstraints);
        fn problem_subject_to_ineq(problem: Pin<&mut Problem>, c: &InequalityConstraints);
        fn problem_cost_function_type(problem: &Problem) -> u8;
        fn problem_equality_constraint_type(problem: &Problem) -> u8;
        fn problem_inequality_constraint_type(problem: &Problem) -> u8;
        fn problem_solve(problem: Pin<&mut Problem>, options: SolverOptions) -> i8;

        fn problem_add_callback(problem: Pin<&mut Problem>, callback: Box<RustCallback>);
        fn problem_add_persistent_callback(
            problem: Pin<&mut Problem>,
            callback: Box<RustCallback>,
        );
        fn problem_clear_callbacks(problem: Pin<&mut Problem>);

        // Gradient
        fn gradient_new(
            variable: &Variable,
            wrt: &VariableMatrix,
        ) -> UniquePtr<Gradient>;
        fn gradient_value(gradient: Pin<&mut Gradient>) -> Vec<f64>;
        fn gradient_get(gradient: &Gradient) -> UniquePtr<VariableMatrix>;

        // Hessian. `uplo_tag`: 0 = lower-only, 1 = full symmetric.
        fn hessian_new(
            variable: &Variable,
            wrt: &VariableMatrix,
            uplo_tag: i32,
        ) -> UniquePtr<Hessian>;
        fn hessian_value(hessian: Pin<&mut Hessian>) -> Vec<f64>;
        fn hessian_rows(hessian: &Hessian) -> i32;
        fn hessian_cols(hessian: &Hessian) -> i32;
        fn hessian_get(hessian: &Hessian) -> UniquePtr<VariableMatrix>;

        // Jacobian
        fn jacobian_new(
            variables: &VariableMatrix,
            wrt: &VariableMatrix,
        ) -> UniquePtr<Jacobian>;
        fn jacobian_value(jacobian: Pin<&mut Jacobian>) -> Vec<f64>;
        fn jacobian_rows(jacobian: &Jacobian) -> i32;
        fn jacobian_cols(jacobian: &Jacobian) -> i32;
        fn jacobian_get(jacobian: &Jacobian) -> UniquePtr<VariableMatrix>;

        // OCP
        fn ocp_new_discrete(
            num_states: i32,
            num_inputs: i32,
            dt_seconds: f64,
            num_steps: i32,
            dynamics: Box<RustDynamics>,
            timestep_method: u8,
        ) -> UniquePtr<OCP>;
        fn ocp_new_explicit_ode(
            num_states: i32,
            num_inputs: i32,
            dt_seconds: f64,
            num_steps: i32,
            dynamics: Box<RustDynamics>,
            timestep_method: u8,
            transcription_method: u8,
        ) -> UniquePtr<OCP>;
        fn ocp_new_discrete_4arg(
            num_states: i32,
            num_inputs: i32,
            dt_seconds: f64,
            num_steps: i32,
            dynamics: Box<RustDynamics4>,
            timestep_method: u8,
        ) -> UniquePtr<OCP>;
        fn ocp_new_explicit_ode_4arg(
            num_states: i32,
            num_inputs: i32,
            dt_seconds: f64,
            num_steps: i32,
            dynamics: Box<RustDynamics4>,
            timestep_method: u8,
            transcription_method: u8,
        ) -> UniquePtr<OCP>;
        fn ocp_constrain_initial_state(ocp: Pin<&mut OCP>, initial: &VariableMatrix);
        fn ocp_constrain_final_state(ocp: Pin<&mut OCP>, final_state: &VariableMatrix);
        fn ocp_set_lower_input_bound_matrix(ocp: Pin<&mut OCP>, bound: &VariableMatrix);
        fn ocp_set_upper_input_bound_matrix(ocp: Pin<&mut OCP>, bound: &VariableMatrix);
        fn ocp_set_min_timestep(ocp: Pin<&mut OCP>, min_timestep_seconds: f64);
        fn ocp_set_max_timestep(ocp: Pin<&mut OCP>, max_timestep_seconds: f64);
        fn ocp_X(ocp: Pin<&mut OCP>) -> UniquePtr<VariableMatrix>;
        fn ocp_U(ocp: Pin<&mut OCP>) -> UniquePtr<VariableMatrix>;
        fn ocp_dt(ocp: Pin<&mut OCP>) -> UniquePtr<VariableMatrix>;
        fn ocp_initial_state(ocp: Pin<&mut OCP>) -> UniquePtr<VariableMatrix>;
        fn ocp_final_state(ocp: Pin<&mut OCP>) -> UniquePtr<VariableMatrix>;
        fn ocp_minimize(ocp: Pin<&mut OCP>, cost: &Variable);
        fn ocp_minimize_matrix(ocp: Pin<&mut OCP>, cost: &VariableMatrix);
        fn ocp_maximize(ocp: Pin<&mut OCP>, objective: &Variable);
        fn ocp_subject_to_eq(ocp: Pin<&mut OCP>, c: &EqualityConstraints);
        fn ocp_subject_to_ineq(ocp: Pin<&mut OCP>, c: &InequalityConstraints);
        fn ocp_solve(ocp: Pin<&mut OCP>, options: SolverOptions) -> i8;
        fn ocp_add_callback(ocp: Pin<&mut OCP>, callback: Box<RustCallback>);
        fn ocp_add_persistent_callback(ocp: Pin<&mut OCP>, callback: Box<RustCallback>);
        fn ocp_clear_callbacks(ocp: Pin<&mut OCP>);
        fn ocp_num_steps(ocp: &OCP) -> i32;
    }
}

pub(crate) use ffi::SolverOptions;

pub struct RustCallback {
    pub(crate) inner: Box<
        dyn for<'a> FnMut(&crate::IterationInfo<'a>) -> bool + Send,
    >,
}

impl RustCallback {
    #[inline]
    fn invoke(&mut self, info: &ffi::IterationInfo) -> bool {
        let view = crate::IterationInfo { inner: info };
        (self.inner)(&view)
    }
}

pub struct RustDynamics {
    pub(crate) inner: Box<
        dyn FnMut(&ffi::VariableMatrix, &ffi::VariableMatrix) -> cxx::UniquePtr<ffi::VariableMatrix>
            + Send,
    >,
}

impl RustDynamics {
    #[inline]
    fn invoke(
        &mut self,
        x: &ffi::VariableMatrix,
        u: &ffi::VariableMatrix,
    ) -> cxx::UniquePtr<ffi::VariableMatrix> {
        (self.inner)(x, u)
    }
}

pub struct RustDynamics4 {
    pub(crate) inner: Box<
        dyn FnMut(
                &ffi::Variable,
                &ffi::VariableMatrix,
                &ffi::VariableMatrix,
                &ffi::Variable,
            ) -> cxx::UniquePtr<ffi::VariableMatrix>
            + Send,
    >,
}

impl RustDynamics4 {
    #[inline]
    fn invoke(
        &mut self,
        t: &ffi::Variable,
        x: &ffi::VariableMatrix,
        u: &ffi::VariableMatrix,
        dt: &ffi::Variable,
    ) -> cxx::UniquePtr<ffi::VariableMatrix> {
        (self.inner)(t, x, u, dt)
    }
}
