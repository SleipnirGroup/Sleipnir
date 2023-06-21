from jormungandr.autodiff import Variable


def test_equality_constraint_boolean_comparisons():
    assert Variable(1.0) == Variable(1.0)
    assert not (Variable(1.0) == Variable(2.0))


def test_inequality_constraint_boolean_comparisons():
    # These are all true because for the purposes of optimization, a <
    # constraint is treated the same as a <= constraint
    assert Variable(1.0) < Variable(1.0)
    assert Variable(1.0) <= Variable(1.0)
    assert Variable(1.0) > Variable(1.0)
    assert Variable(1.0) >= Variable(1.0)

    assert Variable(1.0) < Variable(2.0)
    assert Variable(1.0) <= Variable(2.0)
    assert not (Variable(1.0) > Variable(2.0))
    assert not (Variable(1.0) >= Variable(2.0))

    assert not (Variable(2.0) < Variable(1.0))
    assert not (Variable(2.0) <= Variable(1.0))
    assert Variable(2.0) > Variable(1.0)
    assert Variable(2.0) >= Variable(1.0)
