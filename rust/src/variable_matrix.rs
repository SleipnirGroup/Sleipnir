use std::marker::PhantomData;
use std::ops::{Add, Bound, Div, Mul, Neg, RangeBounds, Sub};
use std::pin::Pin;

use cxx::UniquePtr;
use ndarray::{Array2, ArrayView2};

use crate::arena::VariableArena;
use crate::ffi::ffi;
use crate::variable::__dsl::IntoMatrixOperand;
use crate::variable::{IntoVariable, Variable};

/// A dense matrix of autodiff variables. Row-major storage on the C++ side,
/// matching `slp::VariableMatrix<double>`.
///
/// Every matrix is born in some [`VariableArena`] and propagates that
/// lifetime to every [`Variable`] it hands out via [`Self::get`],
/// [`Self::row`], or arithmetic results. Matrices themselves are
/// `Clone`-not-`Copy` — reuse matrices via `&m` in expressions, or clone
/// (cheap: ref-count bump on the shared expression nodes).
pub struct VariableMatrix<'arena> {
    pub(crate) inner: UniquePtr<ffi::VariableMatrix>,
    pub(crate) arena: &'arena VariableArena,
    _marker: PhantomData<&'arena ()>,
}

impl<'arena> VariableMatrix<'arena> {
    /// Zero-initialised `rows × cols` matrix of fresh decision-variable
    /// expressions, anchored to `arena`.
    pub fn zeros_in(arena: &'arena VariableArena, rows: i32, cols: i32) -> Self {
        Self::from_unique_in(arena, ffi::variable_matrix_zeros(rows, cols))
    }

    /// Constant-valued matrix from an `ndarray::Array2<f64>`.
    pub fn from_array_in(arena: &'arena VariableArena, values: &Array2<f64>) -> Self {
        Self::from_array_view_in(arena, values.view())
    }

    /// Constant-valued matrix from an `ndarray` view.
    pub fn from_array_view_in(arena: &'arena VariableArena, values: ArrayView2<'_, f64>) -> Self {
        let rows = values.nrows() as i32;
        let cols = values.ncols() as i32;
        let flat: Vec<f64> = values.iter().copied().collect();
        Self::from_unique_in(arena, ffi::variable_matrix_from_f64(rows, cols, &flat))
    }

    /// Row-major flat-buffer construction.
    pub fn from_slice_in(arena: &'arena VariableArena, rows: i32, cols: i32, data: &[f64]) -> Self {
        assert_eq!(data.len(), (rows * cols) as usize);
        Self::from_unique_in(arena, ffi::variable_matrix_from_f64(rows, cols, data))
    }

    /// Wraps a single scalar `Variable` in a 1×1 `VariableMatrix`.
    pub fn from_variable_in(arena: &'arena VariableArena, v: Variable<'arena>) -> Self {
        Self::from_unique_in(arena, ffi::variable_matrix_from_variable(v.as_ref()))
    }

    /// `n × n` identity matrix.
    pub fn identity_in(arena: &'arena VariableArena, n: i32) -> Self {
        Self::from_array_in(arena, &Array2::eye(n as usize))
    }

    /// Diagonal matrix with the given entries on the main diagonal.
    /// Off-diagonal elements are zero.
    pub fn diag_in(arena: &'arena VariableArena, diagonal: &[f64]) -> Self {
        let n = diagonal.len() as i32;
        let mut data = vec![0.0_f64; (n * n) as usize];
        for i in 0..n as usize {
            data[i * n as usize + i] = diagonal[i];
        }
        Self::from_slice_in(arena, n, n, &data)
    }

    #[inline]
    pub(crate) fn from_unique_in(
        arena: &'arena VariableArena,
        inner: UniquePtr<ffi::VariableMatrix>,
    ) -> Self {
        Self {
            inner,
            arena,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub(crate) fn as_ref(&self) -> &ffi::VariableMatrix {
        self.inner
            .as_ref()
            .expect("VariableMatrix FFI pointer was unexpectedly null")
    }

    #[inline]
    pub(crate) fn as_pin_mut(&mut self) -> Pin<&mut ffi::VariableMatrix> {
        self.inner
            .as_mut()
            .expect("VariableMatrix FFI pointer was unexpectedly null")
    }

    /// The arena backing this matrix. Useful for building new
    /// matrices/variables next to an existing one when you don't have
    /// the arena reference at hand.
    #[inline]
    pub fn arena(&self) -> &'arena VariableArena {
        self.arena
    }

    pub fn rows(&self) -> i32 {
        ffi::variable_matrix_rows(self.as_ref())
    }

    pub fn cols(&self) -> i32 {
        ffi::variable_matrix_cols(self.as_ref())
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows() as usize, self.cols() as usize)
    }

    /// Element accessor. The returned [`Variable`] borrows the same arena
    /// as this matrix.
    pub fn get(&self, row: i32, col: i32) -> Variable<'arena> {
        Variable::from_unique_in(
            self.arena,
            ffi::variable_matrix_get(self.as_ref(), row, col),
        )
    }

    pub fn value_at(&mut self, row: i32, col: i32) -> f64 {
        ffi::variable_matrix_value_at(self.as_pin_mut(), row, col)
    }

    pub fn set_variable(&mut self, row: i32, col: i32, v: Variable<'arena>) {
        ffi::variable_matrix_set_variable(self.as_pin_mut(), row, col, v.as_ref());
    }

    /// Replaces slot `(row, col)` with a constant expression, severing
    /// any references earlier-built expressions held to the old slot.
    /// For in-place value updates, prefer
    /// [`set_value_at`](Self::set_value_at).
    pub fn set_scalar(&mut self, row: i32, col: i32, value: f64) {
        ffi::variable_matrix_set_f64(self.as_pin_mut(), row, col, value);
    }

    /// Mutates the value of the existing expression at `(row, col)`
    /// in place. Mirrors `slp::Variable::set_value`.
    pub fn set_value_at(&mut self, row: i32, col: i32, value: f64) {
        ffi::variable_matrix_set_value_at(self.as_pin_mut(), row, col, value);
    }

    pub fn value(&mut self) -> Array2<f64> {
        let (rows, cols) = self.shape();
        let flat = ffi::variable_matrix_value(self.as_pin_mut());
        Array2::from_shape_vec((rows, cols), flat)
            .expect("VariableMatrix value buffer shape mismatch")
    }

    pub fn set_value(&mut self, values: &Array2<f64>) {
        let (rows, cols) = self.shape();
        assert_eq!((values.nrows(), values.ncols()), (rows, cols));
        let flat: Vec<f64> = values.iter().copied().collect();
        ffi::variable_matrix_set_value(self.as_pin_mut(), &flat);
    }

    pub fn t(&self) -> VariableMatrix<'arena> {
        VariableMatrix::from_unique_in(self.arena, ffi::variable_matrix_transpose(self.as_ref()))
    }

    pub fn block(
        &self,
        row_offset: i32,
        col_offset: i32,
        block_rows: i32,
        block_cols: i32,
    ) -> VariableMatrix<'arena> {
        VariableMatrix::from_unique_in(
            self.arena,
            ffi::variable_matrix_block(
                self.as_ref(),
                row_offset,
                col_offset,
                block_rows,
                block_cols,
            ),
        )
    }

    /// Range-based block slicing, mirroring NumPy `mat[rows, cols]`
    /// semantics. Accepts any `RangeBounds<i32>` (`a..b`, `a..=b`,
    /// `..b`, `a..`, `..`). Start is inclusive, end exclusive — full
    /// ranges span the whole dimension.
    ///
    /// ```ignore
    /// let sub = mat.slice(1..3, ..);   // rows 1..3, all columns
    /// let top = mat.slice(..=0, 2..);  // row 0, columns 2 to end
    /// ```
    pub fn slice<R, C>(&self, rows: R, cols: C) -> VariableMatrix<'arena>
    where
        R: RangeBounds<i32>,
        C: RangeBounds<i32>,
    {
        let (row_offset, block_rows) = resolve_range(rows, self.rows());
        let (col_offset, block_cols) = resolve_range(cols, self.cols());
        self.block(row_offset, col_offset, block_rows, block_cols)
    }

    pub fn row(&self, row: i32) -> VariableMatrix<'arena> {
        VariableMatrix::from_unique_in(self.arena, ffi::variable_matrix_row(self.as_ref(), row))
    }

    pub fn col(&self, col: i32) -> VariableMatrix<'arena> {
        VariableMatrix::from_unique_in(self.arena, ffi::variable_matrix_col(self.as_ref(), col))
    }

    /// Iterate the matrix's rows as `1 × cols` matrices.
    pub fn rows_iter(&self) -> impl Iterator<Item = VariableMatrix<'arena>> + '_ {
        (0..self.rows()).map(move |r| self.row(r))
    }

    /// Iterate the matrix's columns as `rows × 1` matrices.
    pub fn cols_iter(&self) -> impl Iterator<Item = VariableMatrix<'arena>> + '_ {
        (0..self.cols()).map(move |c| self.col(c))
    }

    /// Element-wise product (Hadamard). Accepts any [`IntoMatrixOperand`].
    pub fn hadamard<R: IntoMatrixOperand<'arena>>(&self, rhs: R) -> VariableMatrix<'arena> {
        let rhs = rhs.into_matrix(self.arena);
        VariableMatrix::from_unique_in(
            self.arena,
            ffi::variable_matrix_hadamard(self.as_ref(), rhs.as_ref()),
        )
    }

    pub fn sum(&self) -> Variable<'arena> {
        Variable::from_unique_in(self.arena, ffi::variable_matrix_sum(self.as_ref()))
    }
}

/// Solve `A * X = B` for `X`, symbolically (differentiable result).
/// The arena comes from `a`.
pub fn solve<'a, A, B>(a: A, b: B) -> VariableMatrix<'a>
where
    A: crate::__marker::HasArena<'a> + IntoMatrixOperand<'a>,
    B: IntoMatrixOperand<'a>,
{
    let arena = a.arena_ref();
    let a = a.into_matrix(arena);
    let b = b.into_matrix(arena);
    VariableMatrix::from_unique_in(arena, ffi::variable_matrix_solve(a.as_ref(), b.as_ref()))
}

/// Vertical stack: concatenate along the row axis. Both operands must
/// have the same column count and share the same arena. The arena is
/// extracted from `top` — which must therefore be a `VariableMatrix`
/// (compile error if it's a bare `ndarray::Array2`).
pub fn vstack<'a, A, B>(top: A, bottom: B) -> VariableMatrix<'a>
where
    A: crate::__marker::HasArena<'a> + IntoMatrixOperand<'a>,
    B: IntoMatrixOperand<'a>,
{
    let arena = top.arena_ref();
    let top = top.into_matrix(arena);
    let bottom = bottom.into_matrix(arena);
    VariableMatrix::from_unique_in(
        arena,
        ffi::variable_matrix_vstack(top.as_ref(), bottom.as_ref()),
    )
}

/// Horizontal stack: concatenate along the column axis. Both operands
/// must have the same row count and share the same arena. The arena is
/// extracted from `left` — which must therefore be a `VariableMatrix`
/// (compile error if it's a bare `ndarray::Array2`).
pub fn hstack<'a, A, B>(left: A, right: B) -> VariableMatrix<'a>
where
    A: crate::__marker::HasArena<'a> + IntoMatrixOperand<'a>,
    B: IntoMatrixOperand<'a>,
{
    let arena = left.arena_ref();
    let left = left.into_matrix(arena);
    let right = right.into_matrix(arena);
    VariableMatrix::from_unique_in(
        arena,
        ffi::variable_matrix_hstack(left.as_ref(), right.as_ref()),
    )
}

/// Resolve `RangeBounds<i32>` into a `(start, length)` tuple against
/// an axis of size `len`. Matches slice::index semantics — panics on
/// out-of-bounds or inverted ranges.
fn resolve_range<R: RangeBounds<i32>>(range: R, len: i32) -> (i32, i32) {
    let start = match range.start_bound() {
        Bound::Included(&s) => s,
        Bound::Excluded(&s) => s + 1,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&e) => e + 1,
        Bound::Excluded(&e) => e,
        Bound::Unbounded => len,
    };
    assert!(
        start <= end,
        "slice start ({start}) must not exceed end ({end})"
    );
    assert!(
        start >= 0 && end <= len,
        "slice range {start}..{end} out of bounds for axis of length {len}"
    );
    (start, end - start)
}

impl<'arena> Clone for VariableMatrix<'arena> {
    fn clone(&self) -> Self {
        VariableMatrix::from_unique_in(self.arena, ffi::variable_matrix_clone(self.as_ref()))
    }
}

impl std::fmt::Debug for VariableMatrix<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VariableMatrix")
            .field("rows", &self.rows())
            .field("cols", &self.cols())
            .finish()
    }
}

macro_rules! impl_matrix_binop {
    ($trait:ident, $method:ident, $ffi:ident) => {
        impl<'a, R: IntoMatrixOperand<'a>> $trait<R> for VariableMatrix<'a> {
            type Output = VariableMatrix<'a>;
            #[inline]
            fn $method(self, rhs: R) -> VariableMatrix<'a> {
                let rhs = rhs.into_matrix(self.arena);
                VariableMatrix::from_unique_in(self.arena, ffi::$ffi(self.as_ref(), rhs.as_ref()))
            }
        }
        impl<'a, R: IntoMatrixOperand<'a>> $trait<R> for &VariableMatrix<'a> {
            type Output = VariableMatrix<'a>;
            #[inline]
            fn $method(self, rhs: R) -> VariableMatrix<'a> {
                let rhs = rhs.into_matrix(self.arena);
                VariableMatrix::from_unique_in(self.arena, ffi::$ffi(self.as_ref(), rhs.as_ref()))
            }
        }
    };
}

impl_matrix_binop!(Add, add, variable_matrix_add);
impl_matrix_binop!(Sub, sub, variable_matrix_sub);
impl_matrix_binop!(Mul, mul, variable_matrix_matmul);

impl<'a, R: IntoVariable<'a>> Div<R> for VariableMatrix<'a> {
    type Output = VariableMatrix<'a>;
    #[inline]
    fn div(self, rhs: R) -> VariableMatrix<'a> {
        let rhs = rhs.into_variable(self.arena);
        VariableMatrix::from_unique_in(
            self.arena,
            ffi::variable_matrix_scalar_div(self.as_ref(), rhs.as_ref()),
        )
    }
}

impl<'a, R: IntoVariable<'a>> Div<R> for &VariableMatrix<'a> {
    type Output = VariableMatrix<'a>;
    #[inline]
    fn div(self, rhs: R) -> VariableMatrix<'a> {
        let rhs = rhs.into_variable(self.arena);
        VariableMatrix::from_unique_in(
            self.arena,
            ffi::variable_matrix_scalar_div(self.as_ref(), rhs.as_ref()),
        )
    }
}

impl<'a> Neg for VariableMatrix<'a> {
    type Output = VariableMatrix<'a>;
    fn neg(self) -> VariableMatrix<'a> {
        VariableMatrix::from_unique_in(self.arena, ffi::variable_matrix_neg(self.as_ref()))
    }
}

impl<'a> Neg for &VariableMatrix<'a> {
    type Output = VariableMatrix<'a>;
    fn neg(self) -> VariableMatrix<'a> {
        VariableMatrix::from_unique_in(self.arena, ffi::variable_matrix_neg(self.as_ref()))
    }
}

// f64 on the LHS — orphan rule forces concrete impls.
impl<'a> Mul<VariableMatrix<'a>> for f64 {
    type Output = VariableMatrix<'a>;
    fn mul(self, rhs: VariableMatrix<'a>) -> VariableMatrix<'a> {
        let lhs = Variable::constant_in(rhs.arena, self);
        VariableMatrix::from_unique_in(
            rhs.arena,
            ffi::variable_matrix_scalar_mul(rhs.as_ref(), lhs.as_ref()),
        )
    }
}

impl<'a> Mul<&VariableMatrix<'a>> for f64 {
    type Output = VariableMatrix<'a>;
    fn mul(self, rhs: &VariableMatrix<'a>) -> VariableMatrix<'a> {
        let lhs = Variable::constant_in(rhs.arena, self);
        VariableMatrix::from_unique_in(
            rhs.arena,
            ffi::variable_matrix_scalar_mul(rhs.as_ref(), lhs.as_ref()),
        )
    }
}

impl<'a> Add<VariableMatrix<'a>> for f64 {
    type Output = VariableMatrix<'a>;
    fn add(self, rhs: VariableMatrix<'a>) -> VariableMatrix<'a> {
        let lhs =
            VariableMatrix::from_variable_in(rhs.arena, Variable::constant_in(rhs.arena, self));
        VariableMatrix::from_unique_in(
            rhs.arena,
            ffi::variable_matrix_add(lhs.as_ref(), rhs.as_ref()),
        )
    }
}

impl<'a> Add<&VariableMatrix<'a>> for f64 {
    type Output = VariableMatrix<'a>;
    fn add(self, rhs: &VariableMatrix<'a>) -> VariableMatrix<'a> {
        let lhs =
            VariableMatrix::from_variable_in(rhs.arena, Variable::constant_in(rhs.arena, self));
        VariableMatrix::from_unique_in(
            rhs.arena,
            ffi::variable_matrix_add(lhs.as_ref(), rhs.as_ref()),
        )
    }
}

impl<'a> Sub<VariableMatrix<'a>> for f64 {
    type Output = VariableMatrix<'a>;
    fn sub(self, rhs: VariableMatrix<'a>) -> VariableMatrix<'a> {
        let lhs =
            VariableMatrix::from_variable_in(rhs.arena, Variable::constant_in(rhs.arena, self));
        VariableMatrix::from_unique_in(
            rhs.arena,
            ffi::variable_matrix_sub(lhs.as_ref(), rhs.as_ref()),
        )
    }
}

impl<'a> Sub<&VariableMatrix<'a>> for f64 {
    type Output = VariableMatrix<'a>;
    fn sub(self, rhs: &VariableMatrix<'a>) -> VariableMatrix<'a> {
        let lhs =
            VariableMatrix::from_variable_in(rhs.arena, Variable::constant_in(rhs.arena, self));
        VariableMatrix::from_unique_in(
            rhs.arena,
            ffi::variable_matrix_sub(lhs.as_ref(), rhs.as_ref()),
        )
    }
}
