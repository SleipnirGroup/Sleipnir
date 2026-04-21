use crate::__marker::HasArena;
use crate::ffi::ffi;
use crate::variable::{IntoVariable, Variable};

// Every math function below takes the arena implicitly via the first
// operand's `HasArena` impl — that's `Variable<'a>` / `&Variable<'a>` /
// `VariableMatrix<'a>` / `&VariableMatrix<'a>`. A bare `f64` / `i32`
// doesn't carry an arena and so isn't accepted as the leading argument;
// do scalar-only math with the `f64` methods (e.g. `(2.0_f64).sin()`).

macro_rules! unary {
    ($(#[$meta:meta])* $rust_name:ident, $ffi_name:ident) => {
        $(#[$meta])*
        #[inline]
        pub fn $rust_name<'a, X>(x: X) -> Variable<'a>
        where
            X: HasArena<'a> + IntoVariable<'a>,
        {
            let arena = x.arena_ref();
            let x = x.into_variable(arena);
            Variable::from_unique_in(arena, ffi::$ffi_name(x.as_ref()))
        }
    };
}

unary!(/// `|x|` for Variables.
       abs, variable_abs);
unary!(/// `acos(x)` for Variables.
       acos, variable_acos);
unary!(/// `asin(x)` for Variables.
       asin, variable_asin);
unary!(/// `atan(x)` for Variables.
       atan, variable_atan);
unary!(/// `cbrt(x)` for Variables.
       cbrt, variable_cbrt);
unary!(/// `cos(x)` for Variables.
       cos, variable_cos);
unary!(/// `cosh(x)` for Variables.
       cosh, variable_cosh);
unary!(/// `erf(x)` for Variables.
       erf, variable_erf);
unary!(/// `exp(x)` for Variables.
       exp, variable_exp);
unary!(/// `log(x)` for Variables.
       log, variable_log);
unary!(/// `log10(x)` for Variables.
       log10, variable_log10);
unary!(/// `sign(x)` for Variables.
       sign, variable_sign);
unary!(/// `sin(x)` for Variables.
       sin, variable_sin);
unary!(/// `sinh(x)` for Variables.
       sinh, variable_sinh);
unary!(/// `sqrt(x)` for Variables.
       sqrt, variable_sqrt);
unary!(/// `tan(x)` for Variables.
       tan, variable_tan);
unary!(/// `tanh(x)` for Variables.
       tanh, variable_tanh);

/// `atan2(y, x)` for Variables. Arena taken from `y`.
#[inline]
pub fn atan2<'a, Y, X>(y: Y, x: X) -> Variable<'a>
where
    Y: HasArena<'a> + IntoVariable<'a>,
    X: IntoVariable<'a>,
{
    let arena = y.arena_ref();
    let y = y.into_variable(arena);
    let x = x.into_variable(arena);
    Variable::from_unique_in(arena, ffi::variable_atan2(y.as_ref(), x.as_ref()))
}

/// `hypot(x, y)` for Variables. Arena taken from `x`.
#[inline]
pub fn hypot<'a, X, Y>(x: X, y: Y) -> Variable<'a>
where
    X: HasArena<'a> + IntoVariable<'a>,
    Y: IntoVariable<'a>,
{
    let arena = x.arena_ref();
    let x = x.into_variable(arena);
    let y = y.into_variable(arena);
    Variable::from_unique_in(arena, ffi::variable_hypot(x.as_ref(), y.as_ref()))
}

/// 3-arg `hypot(x, y, z) = sqrt(x² + y² + z²)` for Variables. Arena
/// taken from `x`.
#[inline]
pub fn hypot3<'a, X, Y, Z>(x: X, y: Y, z: Z) -> Variable<'a>
where
    X: HasArena<'a> + IntoVariable<'a>,
    Y: IntoVariable<'a>,
    Z: IntoVariable<'a>,
{
    let arena = x.arena_ref();
    let x = x.into_variable(arena);
    let y = y.into_variable(arena);
    let z = z.into_variable(arena);
    Variable::from_unique_in(
        arena,
        ffi::variable_hypot3(x.as_ref(), y.as_ref(), z.as_ref()),
    )
}

/// `max(a, b)` for Variables. Arena taken from `a`.
#[inline]
pub fn max<'a, A, B>(a: A, b: B) -> Variable<'a>
where
    A: HasArena<'a> + IntoVariable<'a>,
    B: IntoVariable<'a>,
{
    let arena = a.arena_ref();
    let a = a.into_variable(arena);
    let b = b.into_variable(arena);
    Variable::from_unique_in(arena, ffi::variable_max(a.as_ref(), b.as_ref()))
}

/// `min(a, b)` for Variables. Arena taken from `a`.
#[inline]
pub fn min<'a, A, B>(a: A, b: B) -> Variable<'a>
where
    A: HasArena<'a> + IntoVariable<'a>,
    B: IntoVariable<'a>,
{
    let arena = a.arena_ref();
    let a = a.into_variable(arena);
    let b = b.into_variable(arena);
    Variable::from_unique_in(arena, ffi::variable_min(a.as_ref(), b.as_ref()))
}

/// `pow(base, power)` for Variables. Arena taken from `base`.
#[inline]
pub fn pow<'a, B, P>(base: B, power: P) -> Variable<'a>
where
    B: HasArena<'a> + IntoVariable<'a>,
    P: IntoVariable<'a>,
{
    let arena = base.arena_ref();
    let base = base.into_variable(arena);
    let power = power.into_variable(arena);
    Variable::from_unique_in(
        arena,
        ffi::variable_pow(base.as_ref(), power.as_ref()),
    )
}
