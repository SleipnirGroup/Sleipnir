import concurrent.futures

from .._jormungandr.optimization import *


def multistart(solve, initial_guesses):
    """
    Solves an optimization problem from different starting points in parallel,
    then returns the solution with the lowest cost.

    Each solve is performed on a separate thread. Solutions from successful
    solves are always preferred over solutions from unsuccessful solves, and
    cost (lower is better) is the tiebreaker between successful solves.

    Parameter ``solve``:
        A user-provided function that takes a decision variable initial guess
        and returns a MultistartResult.

    Parameter ``initial_guesses``:
        A list of decision variable initial guesses to try.
    """
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(initial_guesses)
    ) as executor:
        futures = [
            executor.submit(solve, initial_guess) for initial_guess in initial_guesses
        ]
        results = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]

    # Prioritize successful solve, otherwise prioritize solution with lower cost
    return min(
        results,
        key=lambda x: (
            int(x[0] != ExitStatus.SUCCESS),
            x[1],
        ),
    )
