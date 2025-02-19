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

    # Expected values are from the following CasADi program:
    #
    # #!/usr/bin/env python3
    #
    # import casadi as ca
    # import numpy as np
    #
    # opti = ca.Opti()
    # allocated_currents = opti.variable(4, 1)
    #
    # current_tolerances = np.array([[1.0], [5.0], [10.0], [5.0]])
    # desired_currents = np.array([[30.0], [10.0], [5.0], [0.0]])
    #
    # J = 0.0
    # current_sum = 0.0
    # for i in range(4):
    #     error = desired_currents[i, 0] - allocated_currents[i, 0]
    #     J += error**2 / current_tolerances[i] ** 2
    #
    #     current_sum += allocated_currents[i, 0]
    #
    #     # Currents must be nonnegative
    #     opti.subject_to(allocated_currents[i, 0] >= 0.0)
    # opti.minimize(J)
    #
    # # Keep total current below maximum
    # opti.subject_to(current_sum <= 40.0)
    #
    # opti.solver("ipopt")
    # print(opti.solve().value(allocated_currents))
    assert currents[0] == pytest.approx(29.960, abs=1e-3)
    assert currents[1] == pytest.approx(9.007, abs=1e-3)
    assert currents[2] == pytest.approx(1.032, abs=1e-3)
    assert currents[3] == pytest.approx(0.0, abs=1e-3)
