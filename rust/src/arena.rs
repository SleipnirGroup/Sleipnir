use std::sync::Mutex;

use cxx::UniquePtr;

use crate::ffi::ffi;

/// Backing storage for every [`Variable`](crate::Variable) a
/// [`Problem`](crate::Problem) builds.
///
/// The arena owns every FFI wrapper produced during problem construction
/// and frees them in bulk when it drops. Every [`Variable`] borrows from
/// an arena via a `'arena` lifetime, which lets the type be `Copy`
/// without leaking: the borrow checker forces the arena to outlive every
/// handle it produced, so teardown is deterministic instead of global.
///
/// The arena uses an internal `Mutex` so that `&VariableArena` is both
/// `Send` and `Sync`; this is needed because OCP dynamics callbacks are
/// stored behind a `Send`-bounded FFI box. Lock acquisitions happen only
/// on `store()` — i.e. during problem construction, not inside solver
/// iterations — so contention is negligible.
///
/// Typical lifecycle:
///
/// ```ignore
/// use hafgufa::{Problem, VariableArena, subject_to};
///
/// let arena = VariableArena::new();
/// let mut problem = Problem::new(&arena);
/// let x = problem.decision_variable();
/// let y = problem.decision_variable();
/// problem.minimize(&(x * x + y * y));
/// subject_to!(problem, x + y == 1.0);
/// problem.solve(Default::default())?;
/// # Ok::<(), hafgufa::SleipnirError>(())
/// ```
pub struct VariableArena {
    // `UniquePtr<T>` contents live on the heap at a stable address, so
    // the raw pointers we hand out stay valid even when this `Vec`
    // reallocates. `Mutex` gives interior mutability with `Sync`.
    storage: Mutex<Vec<UniquePtr<ffi::Variable>>>,
}

// SAFETY: cxx marks `UniquePtr<T>` `!Send`/`!Sync` by default because the
// pointed-to C++ type may not be thread-safe. For our use:
//
// - Writes to the arena go through `store()`, serialized by the inner
//   `Mutex`.
// - Reads of stored entries happen only by raw pointer (handed out by
//   `store`); no `&mut` or `&UniquePtr<ffi::Variable>` is ever aliased
//   across threads by this type.
// - The only cross-thread touch point is the OCP dynamics closure in
//   `crate::ocp`, which needs `&VariableArena: Send` to satisfy the cxx
//   `Box<RustDynamics>` bound. That closure runs synchronously on the
//   same thread that called `OCP::new_*` — Sleipnir doesn't spawn
//   worker threads during OCP construction — so no race occurs in
//   practice.
//
// Users should not share `Variable<'arena>` handles between threads
// (concurrent `.value()` calls mutate cached graph state inside the C++
// Variable), but that's a separate contract from the arena itself.
unsafe impl Send for VariableArena {}
unsafe impl Sync for VariableArena {}

impl VariableArena {
    /// Creates a new, empty arena.
    pub fn new() -> Self {
        Self {
            storage: Mutex::new(Vec::new()),
        }
    }

    /// Pushes a freshly-constructed FFI wrapper into the arena and
    /// returns a raw pointer that remains valid for the life of `self`.
    pub(crate) fn store(&self, unique: UniquePtr<ffi::Variable>) -> *const ffi::Variable {
        let mut storage = self.storage.lock().expect("VariableArena mutex poisoned");
        let ptr = unique
            .as_ref()
            .expect("Variable FFI pointer was unexpectedly null") as *const _;
        storage.push(unique);
        ptr
    }

    /// Interns a constant `f64` as a fresh [`Variable`](crate::Variable).
    /// Shorthand for `Variable::constant_in(arena, value)`.
    #[inline]
    pub fn constant(&self, value: f64) -> crate::Variable<'_> {
        crate::Variable::constant_in(self, value)
    }

    /// Constant-valued matrix from an `ndarray::Array2<f64>`. Shorthand
    /// for `VariableMatrix::from_array_in(arena, values)`.
    #[inline]
    pub fn array(&self, values: &ndarray::Array2<f64>) -> crate::VariableMatrix<'_> {
        crate::VariableMatrix::from_array_in(self, values)
    }

    /// Constant-valued matrix from an `ndarray` view. Shorthand for
    /// `VariableMatrix::from_array_view_in(arena, values)`.
    #[inline]
    pub fn array_view(&self, values: ndarray::ArrayView2<'_, f64>) -> crate::VariableMatrix<'_> {
        crate::VariableMatrix::from_array_view_in(self, values)
    }

    /// `rows × cols` zero-initialized matrix of fresh decision-variable
    /// expressions. Shorthand for `VariableMatrix::zeros_in(arena, rows, cols)`.
    #[inline]
    pub fn zeros(&self, rows: i32, cols: i32) -> crate::VariableMatrix<'_> {
        crate::VariableMatrix::zeros_in(self, rows, cols)
    }

    /// Row-major flat-buffer matrix construction. Shorthand for
    /// `VariableMatrix::from_slice_in(arena, rows, cols, data)`.
    #[inline]
    pub fn slice(&self, rows: i32, cols: i32, data: &[f64]) -> crate::VariableMatrix<'_> {
        crate::VariableMatrix::from_slice_in(self, rows, cols, data)
    }

    /// `n × n` identity matrix. Shorthand for
    /// `VariableMatrix::identity_in(arena, n)`.
    #[inline]
    pub fn identity(&self, n: i32) -> crate::VariableMatrix<'_> {
        crate::VariableMatrix::identity_in(self, n)
    }

    /// Diagonal matrix with the given entries on the main diagonal.
    /// Shorthand for `VariableMatrix::diag_in(arena, diagonal)`.
    #[inline]
    pub fn diag(&self, diagonal: &[f64]) -> crate::VariableMatrix<'_> {
        crate::VariableMatrix::diag_in(self, diagonal)
    }

    /// Wraps a single [`Variable`](crate::Variable) in a 1×1 matrix.
    /// Shorthand for `VariableMatrix::from_variable_in(arena, v)`.
    #[inline]
    pub fn matrix_from_variable<'a>(&'a self, v: crate::Variable<'a>) -> crate::VariableMatrix<'a> {
        crate::VariableMatrix::from_variable_in(self, v)
    }

    /// Current number of variables the arena has allocated.
    pub fn len(&self) -> usize {
        self.storage
            .lock()
            .expect("VariableArena mutex poisoned")
            .len()
    }

    /// Returns true if no variables have been allocated yet.
    pub fn is_empty(&self) -> bool {
        self.storage
            .lock()
            .expect("VariableArena mutex poisoned")
            .is_empty()
    }
}

impl Default for VariableArena {
    fn default() -> Self {
        Self::new()
    }
}
