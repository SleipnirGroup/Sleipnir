import struct
from typing import BinaryIO

import pytest
from sleipnir.autodiff import ExpressionType
from sleipnir.optimization import ExitStatus, Problem


def read_char(spy: BinaryIO) -> bytes:
    return struct.unpack("c", spy.read(1))[0]


def read_i32(spy: BinaryIO) -> int:
    return struct.unpack("<i", spy.read(4))[0]


def read_str(spy: BinaryIO) -> str:
    size = read_i32(spy)
    return struct.unpack(f"{size}s", spy.read(size))[0].decode("utf-8")


class Coord:
    def __init__(self, row: int, col: int, sign: bytes):
        self.row = row
        self.col = col
        self.sign = sign

    def __eq__(self, other) -> bool:
        return (
            self.row == other.row and self.col == other.col and self.sign == other.sign
        )


def read_coord(spy: BinaryIO) -> Coord:
    return Coord(read_i32(spy), read_i32(spy), read_char(spy))


iterations = 0


def test_problem_spy():
    problem = Problem()

    x, y = problem.decision_variable(2)
    x.set_value(20.0)
    y.set_value(20.0)

    problem.minimize(x**4 + y**4)

    problem.subject_to(x >= 1.0)
    problem.subject_to(x <= 10.0)
    problem.subject_to(y == 2.0)

    global iterations
    iterations = 0

    def callback(info):
        global iterations
        iterations += 1
        return False

    problem.add_callback(callback)

    assert problem.cost_function_type() == ExpressionType.NONLINEAR
    assert problem.equality_constraint_type() == ExpressionType.LINEAR
    assert problem.inequality_constraint_type() == ExpressionType.LINEAR

    assert problem.solve(diagnostics=True, spy=True) == ExitStatus.SUCCESS

    assert x.value() == pytest.approx(1.0, abs=1e-8)
    assert y.value() == pytest.approx(2.0, abs=1e-8)

    # Check H.spy
    with open("H.spy", mode="rb") as spy:
        assert read_str(spy) == "Hessian"  # Title
        assert read_str(spy) == "Decision variables"  # Row label
        assert read_str(spy) == "Decision variables"  # Col label
        assert read_i32(spy) == 2  # Rows
        assert read_i32(spy) == 2  # Cols

        # Coords
        for _ in range(iterations):
            assert read_i32(spy) == 2  # Num coords
            assert read_coord(spy) == Coord(0, 0, b"+")
            assert read_coord(spy) == Coord(1, 1, b"+")

    # Check A_e.spy
    with open("A_e.spy", mode="rb") as spy:
        assert read_str(spy) == "Equality constraint Jacobian"  # Title
        assert read_str(spy) == "Constraints"  # Row label
        assert read_str(spy) == "Decision variables"  # Col label
        assert read_i32(spy) == 1  # Rows
        assert read_i32(spy) == 2  # Cols

        # Coords
        for _ in range(iterations):
            assert read_i32(spy) == 1  # Num coords
            assert read_coord(spy) == Coord(0, 1, b"+")

    # Check A_i.spy
    with open("A_i.spy", mode="rb") as spy:
        assert read_str(spy) == "Inequality constraint Jacobian"  # Title
        assert read_str(spy) == "Constraints"  # Row label
        assert read_str(spy) == "Decision variables"  # Col label
        assert read_i32(spy) == 2  # Rows
        assert read_i32(spy) == 2  # Cols

        # Coords
        for _ in range(iterations):
            assert read_i32(spy) == 2  # Num coords
            assert read_coord(spy) == Coord(0, 0, b"+")
            assert read_coord(spy) == Coord(1, 0, b"-")
