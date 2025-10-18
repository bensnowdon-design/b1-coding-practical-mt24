from __future__ import annotations
import numpy as np


class PDController:
    """Simple discrete-time PD controller.

    u[t] = KP * e[t] + KD * (e[t] - e[t-1])

    KP and KD are configurable; the controller keeps the previous error.
    """

    def __init__(self, kp: float = 0.15, kd: float = 0.6):
        self.kp = kp
        self.kd = kd
        self.prev_error = 0.0

    def reset(self):
        self.prev_error = 0.0

    def control(self, reference: float, observation: float) -> float:
        """Compute control action for current time step.

        Args:
            reference: r[t]
            observation: y[t]

        Returns:
            Control action u[t]
        """
        error = reference - observation
        derivative = error - self.prev_error
        u = self.kp * error + self.kd * derivative
        self.prev_error = error
        return float(u)
