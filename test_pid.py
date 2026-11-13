import numpy as np
import dofbot_controller as controller

controller.reset_pid()

desired_q = np.array([90, 90, 90, 90, 90], dtype=float)
actual_q = np.array([88, 90, 92, 90, 90], dtype=float)

command_q, error, correction = controller.calculate_pid(
    desired_q,
    actual_q,
    dt=0.1
)

assert np.allclose(
    error,
    [2, 0, -2, 0, 0]
)

assert np.allclose(
    correction,
    [1, 0, -1, 0, 0]
)

assert np.allclose(
    command_q,
    [91, 90, 89, 90, 90]
)

print("PASS: PID proportional calculation")