import numpy as np

from jormungandr.optimization import Problem


def test_scalar_init_assign():
    problem = Problem()

    # Scalar zero init
    x = problem.decision_variable()
    assert x.value() == 0.0

    # Scalar assignment
    x.set_value(1.0)
    assert x.value() == 1.0
    x.set_value(2.0)
    assert x.value() == 2.0


def test_vector_init_assign():
    problem = Problem()

    # Vector zero init
    y = problem.decision_variable(2)
    assert y.value(0) == 0.0
    assert y.value(1) == 0.0

    # Vector assignment
    y[0].set_value(1.0)
    y[1].set_value(2.0)
    assert y.value(0) == 1.0
    assert y.value(1) == 2.0
    y[0].set_value(3.0)
    y[1].set_value(4.0)
    assert y.value(0) == 3.0
    assert y.value(1) == 4.0


def test_matrix_init_assign():
    problem = Problem()

    # Matrix zero init
    z = problem.decision_variable(3, 2)
    assert z.value(0, 0) == 0.0
    assert z.value(0, 1) == 0.0
    assert z.value(1, 0) == 0.0
    assert z.value(1, 1) == 0.0
    assert z.value(2, 0) == 0.0
    assert z.value(2, 1) == 0.0

    # Matrix assignment; element comparison
    z.set_value(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    assert z.value(0, 0) == 1.0
    assert z.value(0, 1) == 2.0
    assert z.value(1, 0) == 3.0
    assert z.value(1, 1) == 4.0
    assert z.value(2, 0) == 5.0
    assert z.value(2, 1) == 6.0

    # Matrix assignment; matrix comparison
    z.set_value(np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]))
    assert (z.value() == np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])).all()

    # Block assignment
    z[:2, :1].set_value(np.array([[1.0], [1.0]]))
    assert (z.value() == np.array([[1.0, 8.0], [1.0, 10.0], [11.0, 12.0]])).all()


def test_symmetric_matrix():
    problem = Problem()

    # Matrix zero init
    A = problem.symmetric_decision_variable(2)
    assert A.value(0, 0) == 0.0
    assert A.value(0, 1) == 0.0
    assert A.value(1, 0) == 0.0
    assert A.value(1, 1) == 0.0

    # Assign to lower triangle
    A[0, 0].set_value(1.0)
    A[1, 0].set_value(2.0)
    A[1, 1].set_value(3.0)

    # Confirm whole matrix changed
    assert A.value(0, 0) == 1.0
    assert A.value(0, 1) == 2.0
    assert A.value(1, 0) == 2.0
    assert A.value(1, 1) == 3.0
