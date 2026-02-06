import pytest
from CurrentManager import CurrentManager


def test_current_manager_enough_current():
    manager = CurrentManager([1.0, 5.0, 10.0, 5.0], 40.0)

    currents = manager.calculate([25.0, 10.0, 5.0, 0.0])

    assert currents[0] == pytest.approx(25.0, abs=1e-3)
    assert currents[1] == pytest.approx(10.0, abs=1e-3)
    assert currents[2] == pytest.approx(5.0, abs=1e-3)
    assert currents[3] == pytest.approx(0.0, abs=1e-3)


def test_current_manager_not_enough_current():
    manager = CurrentManager([1.0, 5.0, 10.0, 5.0], 40.0)

    currents = manager.calculate([30.0, 10.0, 5.0, 0.0])

    # Expected values are from the following program:
    #
    # #!/usr/bin/env python3
    #
    # from scipy.optimize import minimize
    #
    # r = [30.0, 10.0, 5.0, 0.0]
    # q = [1.0, 5.0, 10.0, 5.0]
    #
    # result = minimize(
    #     lambda x: sum((r[i] - x[i]) ** 2 / q[i] ** 2 for i in range(4)),
    #     [0.0, 0.0, 0.0, 0.0],
    #     constraints=[
    #         {"type": "ineq", "fun": lambda x: x},
    #         {"type": "ineq", "fun": lambda x: 40.0 - sum(x)},
    #     ],
    # )
    # print(result.x)
    assert currents[0] == pytest.approx(29.960, abs=1e-3)
    assert currents[1] == pytest.approx(9.008, abs=1e-3)
    assert currents[2] == pytest.approx(1.032, abs=1e-3)
    assert currents[3] == pytest.approx(0.0, abs=1e-3)
