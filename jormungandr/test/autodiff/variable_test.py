from jormungandr.autodiff import ExpressionType, Variable


def test_default_constructor():
    a = Variable()

    assert a.value() == 0.0
    assert a.type() == ExpressionType.LINEAR
