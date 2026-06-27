/// Build a constraint from a `lhs OP rhs` expression where `OP` is one
/// of `==`, `<=`, `>=`, `<`, `>`.
///
/// Acts as the standalone equivalent of the relational half of
/// [`subject_to!`] — same tt-muncher pattern, same `SleipnirOperand`
/// static checks, but without a problem to attach the constraint to.
/// The arena is extracted from the left operand; the left operand must
/// therefore be a `Variable` / `VariableMatrix` (compile error if it's a
/// scalar like `f64`).
///
/// ```ignore
/// let c = cmp!(x + y == 1.0);
/// problem.subject_to(c);
///
/// let ineq = cmp!(x >= 0.0);
/// problem.subject_to(ineq);
/// ```
#[macro_export]
macro_rules! cmp {
    ($($rest:tt)*) => {
        $crate::__cmp_munch!(@scan [] $($rest)*)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __cmp_munch {
    (@scan [$($lhs:tt)+] == $($rhs:tt)+) => {{
        let __lhs = { $($lhs)+ };
        let __rhs = { $($rhs)+ };
        $crate::__marker::assert_operand(&__lhs);
        $crate::__marker::assert_operand(&__rhs);
        let __arena = $crate::__marker::arena_from(&__lhs);
        $crate::variable::__dsl::eq_mat(__arena, __lhs, __rhs)
    }};
    (@scan [$($lhs:tt)+] >= $($rhs:tt)+) => {{
        let __lhs = { $($lhs)+ };
        let __rhs = { $($rhs)+ };
        $crate::__marker::assert_operand(&__lhs);
        $crate::__marker::assert_operand(&__rhs);
        let __arena = $crate::__marker::arena_from(&__lhs);
        $crate::variable::__dsl::ge_mat(__arena, __lhs, __rhs)
    }};
    (@scan [$($lhs:tt)+] <= $($rhs:tt)+) => {{
        let __lhs = { $($lhs)+ };
        let __rhs = { $($rhs)+ };
        $crate::__marker::assert_operand(&__lhs);
        $crate::__marker::assert_operand(&__rhs);
        let __arena = $crate::__marker::arena_from(&__lhs);
        $crate::variable::__dsl::le_mat(__arena, __lhs, __rhs)
    }};
    (@scan [$($lhs:tt)+] > $($rhs:tt)+) => {{
        let __lhs = { $($lhs)+ };
        let __rhs = { $($rhs)+ };
        $crate::__marker::assert_operand(&__lhs);
        $crate::__marker::assert_operand(&__rhs);
        let __arena = $crate::__marker::arena_from(&__lhs);
        $crate::variable::__dsl::ge_mat(__arena, __lhs, __rhs)
    }};
    (@scan [$($lhs:tt)+] < $($rhs:tt)+) => {{
        let __lhs = { $($lhs)+ };
        let __rhs = { $($rhs)+ };
        $crate::__marker::assert_operand(&__lhs);
        $crate::__marker::assert_operand(&__rhs);
        let __arena = $crate::__marker::arena_from(&__lhs);
        $crate::variable::__dsl::le_mat(__arena, __lhs, __rhs)
    }};
    (@scan [$($lhs:tt)*] $next:tt $($rest:tt)*) => {
        $crate::__cmp_munch!(@scan [$($lhs)* $next] $($rest)*)
    };
}

/// Add a constraint to a `Problem` / `OCP` using natural mathematical
/// notation.
///
/// The problem container is statically checked against `SleipnirProblem`
/// (only `Problem` and `OCP` qualify) and each operand against
/// `SleipnirOperand`. The arena is extracted from the problem itself, so
/// the user never has to pass it explicitly in this form.
#[macro_export]
macro_rules! subject_to {
    ($problem:expr, $($rest:tt)*) => {
        $crate::__subject_to_munch!(@scan $problem, [] $($rest)*)
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __subject_to_munch {
    (@scan $problem:expr, [$($lhs:tt)+] == $($rhs:tt)+) => {{
        let __lhs = { $($lhs)+ };
        let __rhs = { $($rhs)+ };
        $crate::__marker::assert_problem(&mut $problem);
        $crate::__marker::assert_operand(&__lhs);
        $crate::__marker::assert_operand(&__rhs);
        let __arena = $problem.arena();
        let __c = $crate::variable::__dsl::eq_mat(__arena, __lhs, __rhs);
        $problem.subject_to_equality(__c)
    }};
    (@scan $problem:expr, [$($lhs:tt)+] >= $($rhs:tt)+) => {{
        let __lhs = { $($lhs)+ };
        let __rhs = { $($rhs)+ };
        $crate::__marker::assert_problem(&mut $problem);
        $crate::__marker::assert_operand(&__lhs);
        $crate::__marker::assert_operand(&__rhs);
        let __arena = $problem.arena();
        let __c = $crate::variable::__dsl::ge_mat(__arena, __lhs, __rhs);
        $problem.subject_to_inequality(__c)
    }};
    (@scan $problem:expr, [$($lhs:tt)+] <= $($rhs:tt)+) => {{
        let __lhs = { $($lhs)+ };
        let __rhs = { $($rhs)+ };
        $crate::__marker::assert_problem(&mut $problem);
        $crate::__marker::assert_operand(&__lhs);
        $crate::__marker::assert_operand(&__rhs);
        let __arena = $problem.arena();
        let __c = $crate::variable::__dsl::le_mat(__arena, __lhs, __rhs);
        $problem.subject_to_inequality(__c)
    }};
    (@scan $problem:expr, [$($lhs:tt)+] > $($rhs:tt)+) => {{
        let __lhs = { $($lhs)+ };
        let __rhs = { $($rhs)+ };
        $crate::__marker::assert_problem(&mut $problem);
        $crate::__marker::assert_operand(&__lhs);
        $crate::__marker::assert_operand(&__rhs);
        let __arena = $problem.arena();
        let __c = $crate::variable::__dsl::ge_mat(__arena, __lhs, __rhs);
        $problem.subject_to_inequality(__c)
    }};
    (@scan $problem:expr, [$($lhs:tt)+] < $($rhs:tt)+) => {{
        let __lhs = { $($lhs)+ };
        let __rhs = { $($rhs)+ };
        $crate::__marker::assert_problem(&mut $problem);
        $crate::__marker::assert_operand(&__lhs);
        $crate::__marker::assert_operand(&__rhs);
        let __arena = $problem.arena();
        let __c = $crate::variable::__dsl::le_mat(__arena, __lhs, __rhs);
        $problem.subject_to_inequality(__c)
    }};
    (@scan $problem:expr, [$($lhs:tt)*] $next:tt $($rest:tt)*) => {
        $crate::__subject_to_munch!(@scan $problem, [$($lhs)* $next] $($rest)*)
    };
    (@scan $problem:expr, [$($tokens:tt)+]) => {{
        $crate::__marker::assert_problem(&mut $problem);
        $problem.subject_to({ $($tokens)+ })
    }};
}
