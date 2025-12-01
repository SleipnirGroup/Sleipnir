from sleipnir.autodiff import ExpressionType, Variable


def test_default_constructor():
    a = Variable()
    assert a.value() == 0.0
    assert a.type() == ExpressionType.LINEAR


def test_constant_constructor():
    # float
    a = Variable(1.0)
    assert a.value() == 1.0
    assert a.type() == ExpressionType.CONSTANT

    # int
    b = Variable(2)
    assert b.value() == 2
    assert b.type() == ExpressionType.CONSTANT


def test_set_value():
    a = Variable()

    # float
    a.set_value(1.0)
    assert a.value() == 1.0

    # int
    a.set_value(2)
    assert a.value() == 2
