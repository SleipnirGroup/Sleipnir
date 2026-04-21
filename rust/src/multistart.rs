use rayon::prelude::*;

use crate::{ExitStatus, SleipnirError};

/// Result of a single solve inside [`multistart`].
#[derive(Clone, Debug)]
pub struct MultistartResult<T> {
    /// The exit status from that solve.
    pub status: ExitStatus,
    /// The cost at the returned iterate.
    pub cost: f64,
    /// The caller-supplied payload (typically decoded decision-variable values).
    pub variables: T,
}

/// Solves the same problem from multiple initial guesses in parallel and
/// returns the solution with the lowest cost, preferring successful exits
/// over failed ones.
///
/// Mirrors `slp::multistart` in C++, with the C++ version's `std::async`
/// thread-per-task model replaced by Rust's rayon thread pool. The rayon
/// thread pool is sized once by the rayon runtime; `multistart` itself is
/// cheap to call repeatedly.
///
/// The solve function receives each initial guess and returns a
/// [`MultistartResult`]. Solve functions that return an [`Err`] are folded
/// into a `status = …`, `cost = f64::INFINITY` entry so the best-known
/// solution still wins tie-breaks.
///
/// # Parameters
/// - `initial_guesses`: one parallel solve is spawned per element.
/// - `solve`: called once per initial guess. Must be `Send + Sync` because
///   rayon executes calls on worker threads.
///
/// # Returns
/// The [`MultistartResult`] with the best outcome. `Err` if every solve
/// failed.
pub fn multistart<G, T, F>(
    initial_guesses: &[G],
    solve: F,
) -> Result<MultistartResult<T>, SleipnirError>
where
    G: Sync,
    T: Send,
    F: Fn(&G) -> Result<MultistartResult<T>, SleipnirError> + Sync + Send,
{
    let results: Vec<Result<MultistartResult<T>, SleipnirError>> = initial_guesses
        .par_iter()
        .map(|g| solve(g))
        .collect();

    // Partition into successful and failed solves. A "successful" solve is
    // one whose inner status is `ExitStatus::Success`; anything that
    // returned a non-SUCCESS status (including Err) gets the tiebreaker.
    let mut best: Option<MultistartResult<T>> = None;
    let mut last_err: Option<SleipnirError> = None;

    for result in results {
        match result {
            Ok(r) => best = pick_better(best, r),
            Err(e) => last_err = Some(e),
        }
    }

    best.ok_or_else(|| last_err.unwrap_or(SleipnirError::LocallyInfeasible))
}

fn pick_better<T>(
    best: Option<MultistartResult<T>>,
    candidate: MultistartResult<T>,
) -> Option<MultistartResult<T>> {
    match best {
        None => Some(candidate),
        Some(current) => {
            // Prefer successful over unsuccessful, otherwise lower cost.
            let current_ok = current.status == ExitStatus::Success;
            let cand_ok = candidate.status == ExitStatus::Success;
            if cand_ok && !current_ok {
                Some(candidate)
            } else if !cand_ok && current_ok {
                Some(current)
            } else if candidate.cost < current.cost {
                Some(candidate)
            } else {
                Some(current)
            }
        }
    }
}
