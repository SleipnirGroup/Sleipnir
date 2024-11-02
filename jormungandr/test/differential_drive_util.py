import math

import numpy as np

import jormungandr.autodiff as autodiff
from jormungandr.autodiff import VariableMatrix

# x = [x, y, heading, left velocity, right velocity]ᵀ
# u = [left voltage, right voltage]ᵀ
trackwidth = 0.699  # m
Kv_linear = 3.02  # V/(m/s)
Ka_linear = 0.642  # V/(m/s²)
Kv_angular = 1.382  # V/(m/s)
Ka_angular = 0.08495  # V/(m/s²)

A1 = -(Kv_linear / Ka_linear + Kv_angular / Ka_angular) / 2.0
A2 = -(Kv_linear / Ka_linear - Kv_angular / Ka_angular) / 2.0
B1 = 0.5 / Ka_linear + 0.5 / Ka_angular
B2 = 0.5 / Ka_linear - 0.5 / Ka_angular
A = np.array([[A1, A2], [A2, A1]])
B = np.array([[B1, B2], [B2, B1]])


def differential_drive_dynamics_double(x, u):
    xdot = np.empty((5, 1))

    v = (x[3, 0] + x[4, 0]) / 2.0
    xdot[0, :] = v * math.cos(x[2, 0])
    xdot[1, :] = v * math.sin(x[2, 0])
    xdot[2, :] = (x[4, 0] - x[3, 0]) / trackwidth
    xdot[3:5, :] = A @ x[3:5, :] + B @ u

    return xdot


def differential_drive_dynamics(x, u):
    xdot = VariableMatrix(5, 1)

    v = (x[3, 0] + x[4, 0]) / 2.0
    xdot[0, :] = v * autodiff.cos(x[2, 0])
    xdot[1, :] = v * autodiff.sin(x[2, 0])
    xdot[2, :] = (x[4, 0] - x[3, 0]) / trackwidth
    xdot[3:5, :] = A @ x[3:5, :] + B @ u

    return xdot
