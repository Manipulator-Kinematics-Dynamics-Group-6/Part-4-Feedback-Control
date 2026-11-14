# ==========================================
# DOFBOT ROBOT CONTROLLER
# ==========================================
# Handles joint and gripper commands, trajectory execution,
# optional joint-space PID feedback, and simulation.
# TEST=True uses a configurable first-order simulated servo response.
# TEST=False uses the physical DOFBOT through Arm_Lib.
# Settings and gain values are defined in constants.py.
# ==========================================

import logging
import time
import numpy as np

import constants as const
import dofbot_kinematics as kinematics


# ==========================================
# SETUP LOGGING
# ==========================================

logger = logging.getLogger("dofbot")


# ==========================================
# SIMULATED ROBOT STATE & SERVO DYNAMICS
# ==========================================

# These are internal to this module, not stored in constants.py.
simulated_q = np.array([90, 90, 90, 90, 90], dtype=float)
simulated_target_q = simulated_q.copy()
simulated_gripper = float(const.GRIPPER_OPEN)
simulated_last_update = time.monotonic()


# Resets the simulated arm and target to a known configuration.
def reset_simulation(joint_angles=None, gripper_angle=None):
    global simulated_q, simulated_target_q, simulated_gripper, simulated_last_update

    if not const.TEST:
        raise RuntimeError("Simulation reset is only available when TEST=True.")

    if joint_angles is None:
        joint_angles = [90, 90, 90, 90, 90]
    angles = np.asarray(joint_angles, dtype=float).flatten()
    if angles.shape != (5,) or not np.all(np.isfinite(angles)):
        raise ValueError("Simulation reset requires five finite joint angles.")

    simulated_q = angles.copy()
    simulated_target_q = angles.copy()
    simulated_gripper = float(
        const.GRIPPER_OPEN if gripper_angle is None else gripper_angle
    )
    simulated_last_update = time.monotonic()


# Advances the simulated joints according to elapsed real time.
def update_simulated_joints():
    global simulated_q, simulated_last_update

    if not const.TEST:
        return

    now = time.monotonic()
    dt = max(0.0, now - simulated_last_update)
    simulated_last_update = now

    if not getattr(const, "SIMULATE_SERVO_LAG", False):
        simulated_q = simulated_target_q.copy()
        return

    tau = float(const.SIM_SERVO_TAU)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("SIM_SERVO_TAU must be positive and finite.")

    alpha = 1.0 - np.exp(-dt / tau)
    simulated_q += alpha * (simulated_target_q - simulated_q)


# ==========================================
# HARDWARE CONNECTION
# ==========================================

if not const.TEST:
    from Arm_Lib import Arm_Device
    arm = Arm_Device()


# ==========================================
# JOINT CONTROL
# ==========================================

# Waits for all five joints to reach their target within tolerance.
def wait_for_joints(target_angles, visualizer=None):
    target_angles = np.asarray(target_angles, dtype=float).flatten()
    if target_angles.shape != (5,) or not np.all(np.isfinite(target_angles)):
        raise ValueError("Target must contain exactly five finite joint angles.")

    start_time = time.monotonic()
    while True:
        current_angles = np.asarray(get_joint_angles(), dtype=float)
        if current_angles.shape != (5,) or not np.all(np.isfinite(current_angles)):
            raise RuntimeError("Invalid joint feedback received.")

        error = np.abs(target_angles - current_angles)
        if np.all(error <= const.JOINT_POSITION_TOL):
            logger.debug("Joints reached target | actual=%s | error=%s",
                         current_angles.tolist(), error.tolist())
            return True

        if time.monotonic() - start_time >= const.JOINT_POSITION_TIMEOUT:
            logger.error("Joint timeout | target=%s | actual=%s | error=%s",
                         target_angles.tolist(), current_angles.tolist(), error.tolist())
            return False

        if visualizer is not None:
            visualizer.process_events()
        time.sleep(const.JOINT_FEEDBACK_INTERVAL)


# Sends joint and gripper angles to the simulated or physical arm.
def command_robot(joint_angles, gripper_angle, movement_time=2500):
    global simulated_target_q, simulated_gripper

    joint_angles = np.asarray(joint_angles, dtype=float).flatten()
    if joint_angles.shape != (5,) or not np.all(np.isfinite(joint_angles)):
        raise ValueError("Joint command must contain exactly five finite angles.")

    logger.debug("Joint command | q=%s | gripper=%.2f | ms=%s",
                 joint_angles.tolist(), float(gripper_angle), movement_time)

    if const.TEST:
        # Advance to 'now' before changing the requested servo positions.
        update_simulated_joints()
        simulated_target_q = joint_angles.copy()
        simulated_gripper = float(gripper_angle)
    else:
        arm.Arm_serial_servo_write6(
            *joint_angles, gripper_angle, movement_time
        )


# Reads current joint positions, updating simulated dynamics if needed.
def get_joint_angles():
    if const.TEST:
        update_simulated_joints()
        return simulated_q.copy()

    time.sleep(0.01)
    angles = []
    for joint in range(1, 6):
        angle = arm.Arm_serial_servo_read(joint)
        if angle is None:
            raise RuntimeError(f"Could not read servo {joint}.")
        angles.append(angle)
    return np.asarray(angles, dtype=float)


# ==========================================
# GRIPPER CONTROL
# ==========================================

# Waits for the gripper to reach its requested position.
def wait_for_gripper(target_angle, visualizer=None):
    start_time = time.monotonic()
    while True:
        actual = get_gripper_angle()
        error = abs(target_angle - actual)
        if error <= const.GRIPPER_POSITION_TOL:
            logger.debug("Gripper reached target | actual=%.2f | error=%.2f", actual, error)
            return True

        if time.monotonic() - start_time >= const.GRIPPER_POSITION_TIMEOUT:
            logger.error("Gripper timeout | target=%.2f | actual=%.2f | error=%.2f",
                         target_angle, actual, error)
            return False

        if visualizer is not None:
            visualizer.process_events()
        time.sleep(const.GRIPPER_FEEDBACK_INTERVAL)


# Commands the gripper and verifies reported position.
def command_gripper(angle, movement_time=1000, visualizer=None):
    global simulated_gripper
    if not np.isfinite(angle) or not 0 <= angle <= 180:
        raise ValueError("Gripper angle must be between 0 and 180 degrees.")

    logger.debug("Gripper command | target=%.2f | ms=%s", angle, movement_time)
    if const.TEST:
        simulated_gripper = float(angle)
    else:
        arm.Arm_serial_servo_write(const.GRIPPER_ID, angle, movement_time)

    if not wait_for_gripper(angle, visualizer):
        raise TimeoutError("Gripper failed to reach its target.")


# Reads current gripper angle.
def get_gripper_angle():
    if const.TEST:
        return simulated_gripper

    time.sleep(0.01)
    angle = arm.Arm_serial_servo_read(const.GRIPPER_ID)
    if angle is None:
        raise RuntimeError("Could not read gripper position.")
    return angle


# ==========================================
# PID FEEDBACK CONTROL
# ==========================================

pid_integral = np.zeros(5, dtype=float)
pid_previous_actual = None


# Resets PID state at the start of each movement.
def reset_pid():
    global pid_integral, pid_previous_actual
    pid_integral = np.zeros(5, dtype=float)
    pid_previous_actual = None


# Returns corrected command, tracking error, and correction for five joints.
def calculate_pid(desired_q, actual_q, dt):
    global pid_integral, pid_previous_actual

    desired_q = np.asarray(desired_q, dtype=float).flatten()
    actual_q = np.asarray(actual_q, dtype=float).flatten()
    if desired_q.shape != (5,) or actual_q.shape != (5,):
        raise ValueError("PID requires exactly five joint angles.")
    if not np.all(np.isfinite(desired_q)) or not np.all(np.isfinite(actual_q)):
        raise ValueError("PID joint angles must be finite.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("PID time step must be positive and finite.")

    error = desired_q - actual_q
    P = const.KP * error

    candidate_integral = np.clip(
        pid_integral + error * dt,
        -const.PID_INTEGRAL_LIMIT, const.PID_INTEGRAL_LIMIT
    )

    # Derivative on measured position avoids reference-step kick.
    if pid_previous_actual is None:
        derivative = np.zeros(5, dtype=float)
    else:
        derivative = -(actual_q - pid_previous_actual) / dt
    D = const.KD * derivative

    # Conditional integration limits windup at correction saturation.
    proposed = P + const.KI * candidate_integral + D
    limit = const.PID_MAX_CORRECTION
    winding_up = ((proposed > limit) & (error > 0)) | (
        (proposed < -limit) & (error < 0)
    )
    pid_integral = np.where(winding_up, pid_integral, candidate_integral)

    correction = np.clip(P + const.KI * pid_integral + D, -limit, limit)
    command_q = np.clip(desired_q + correction, const.JOINT_MIN, const.JOINT_MAX)
    pid_previous_actual = actual_q.copy()

    logger.debug("PID | error=%s | correction=%s | command=%s",
                 error.tolist(), correction.tolist(), command_q.tolist())
    return command_q, error, correction


# ==========================================
# EXECUTE TRAJECTORY
# ==========================================

# Executes the current trajectory with optional PID and shared tracking feedback.
def execute_trajectory(gripper_state, visualizer=None, tracking=None):
    if const.N < 2 or const.TOTAL_TIME <= 0:
        raise ValueError("N must be >=2 and TOTAL_TIME must be positive.")

    dt = const.TOTAL_TIME / (const.N - 1)
    movement_time = max(100, int(dt * 1000))
    feedback_interval = const.PID_UPDATE_INTERVAL
    if not isinstance(feedback_interval, (int, np.integer)) or feedback_interval < 1:
        raise ValueError("PID_UPDATE_INTERVAL must be a positive integer.")
    if tracking is not None and const.TRACKING_INTERVAL != feedback_interval:
        raise ValueError("TRACKING_INTERVAL must match PID_UPDATE_INTERVAL.")

    reset_pid()
    correction_q = np.zeros(5, dtype=float)
    start_time = time.monotonic()
    last_feedback_time = None
    paused_duration = 0.0
    logger.debug("Trajectory started | N=%d | duration=%.2fs | PID=%s | lag=%s",
                 const.N, const.TOTAL_TIME, const.PID_ENABLED,
                 const.TEST and getattr(const, "SIMULATE_SERVO_LAG", False))

    try:
        for ii in range(const.N):
            # ==================================
            # PROCESS VISUALIZER AND TIMING
            # ==================================
            if visualizer is not None:
                visualizer.process_events()
                paused_duration += visualizer.wait_if_paused()

            target_time = start_time + ii * dt + paused_duration
            while True:
                if visualizer is not None:
                    visualizer.process_events()
                    pause_time = visualizer.wait_if_paused()
                    if pause_time > 0:
                        paused_duration += pause_time
                        target_time += pause_time

                remaining = target_time - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.01))

            # ==================================
            # DESIRED JOINT CONFIGURATION
            # ==================================
            desired_q = kinematics.q[:, ii].copy()
            feedback_due = (ii % feedback_interval == 0 or ii == const.N - 1)

            # ==================================
            # SHARED FEEDBACK UPDATE
            # ==================================
            if feedback_due and (const.PID_ENABLED or tracking is not None):
                actual_q = get_joint_angles()
                feedback_time = time.monotonic()

                # Align desired position with the instant feedback was sampled.
                elapsed = feedback_time - start_time - paused_duration
                progress = np.clip(elapsed / const.TOTAL_TIME, 0.0, 1.0)
                desired_feedback_q = np.array([
                    np.interp(progress, const.lam, kinematics.q[j, :])
                    for j in range(5)
                ])

                dt_feedback = (dt if last_feedback_time is None
                               else feedback_time - last_feedback_time)
                last_feedback_time = feedback_time

                if const.PID_ENABLED:
                    corrected, error, _ = calculate_pid(
                        desired_feedback_q, actual_q, dt_feedback
                    )
                    correction_q = corrected - desired_feedback_q
                    logger.debug("PID frame %d | error=%s | effective correction=%s",
                                 ii, error.tolist(), correction_q.tolist())

                if tracking is not None:
                    tracking.record(
                        desired_feedback_q, actual_q,
                        measurement_time=feedback_time
                    )

            # ==================================
            # COMMAND ROBOT
            # ==================================
            if const.PID_ENABLED:
                command_q = np.clip(
                    desired_q + correction_q, const.JOINT_MIN, const.JOINT_MAX
                )
            else:
                command_q = desired_q.copy()

            command_robot(command_q, gripper_state, movement_time)
            if visualizer is not None:
                visualizer.update(ii)

            logger.debug("Frame %d/%d | t=%.2fs | desired=%s | commanded=%s",
                         ii + 1, const.N, ii * dt,
                         desired_q.tolist(), command_q.tolist())

        # ==================================
        # VERIFY FINAL JOINT POSITIONS
        # ==================================
        target_angles = kinematics.q[:, -1].copy()
        logger.debug("Trajectory sent; checking final joint positions")
        if not wait_for_joints(target_angles, visualizer):
            raise TimeoutError("Robot failed to reach the final joint configuration.")
        logger.debug("Trajectory complete | elapsed=%.3fs",
                     time.monotonic() - start_time)

    except KeyboardInterrupt:
        logger.warning("Trajectory interrupted by user")
        raise
    except Exception:
        logger.exception("Trajectory execution failed")
        raise
    finally:
        logger.debug("Controller trajectory execution finished")


# ==========================================
# TEST CONFIGURATIONS
# ==========================================

# Runs controller assertions in simulation only.
def test_config():
    assert const.TEST, "Controller tests require TEST=True (no physical movement)."
    qstart = np.array([90, 90, 90, 90, 90], dtype=float)
    qend = np.array([45, 120, 140, 100, 95], dtype=float)

    print("\n==========================================")
    print("DOFBOT CONTROLLER TEST")
    print("==========================================")

    print("\nTEST 1: Initial state")
    reset_simulation(qstart, const.GRIPPER_OPEN)
    assert np.allclose(get_joint_angles(), qstart)
    assert np.isclose(get_gripper_angle(), const.GRIPPER_OPEN)
    print("PASS: Initial state")

    print("\nTEST 2: Joint command updates target")
    command_robot(qend, const.GRIPPER_OPEN)
    assert np.allclose(simulated_target_q, qend)
    if not getattr(const, "SIMULATE_SERVO_LAG", False):
        assert np.allclose(get_joint_angles(), qend)
    print("PASS: Joint target command")

    print("\nTEST 3: Joint position feedback / convergence")
    assert wait_for_joints(qend), "Simulated joints failed to reach target."
    assert np.all(np.abs(get_joint_angles() - qend) <= const.JOINT_POSITION_TOL)
    print("PASS: Joint feedback")

    print("\nTEST 4: Trajectory generation")
    kinematics.generate_trajectory(qstart, qend)
    assert kinematics.q.shape == (5, const.N)
    assert np.allclose(kinematics.q[:, 0], qstart)
    assert np.allclose(kinematics.q[:, -1], qend)
    assert np.all(np.isfinite(kinematics.q))
    print("PASS: Trajectory generation")

    print("\nTEST 5: Trajectory execution")
    reset_simulation(qstart, const.GRIPPER_OPEN)
    execute_trajectory(const.GRIPPER_OPEN)
    assert np.all(np.abs(get_joint_angles() - qend) <= const.JOINT_POSITION_TOL)
    print("PASS: Trajectory execution")

    print("\nTEST 6: Forward kinematics")
    Rot, Pot = kinematics.fwdkin_Dofbot(get_joint_angles())
    assert Rot.shape == (3, 3) and Pot.shape == (3,)
    assert np.allclose(Rot.T @ Rot, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(Rot), 1.0, atol=1e-6)
    assert np.all(np.isfinite(Pot))
    print("PASS: Forward kinematics")

    print("\nTEST 7: Gripper close")
    command_gripper(const.GRIPPER_CLOSED)
    assert abs(get_gripper_angle() - const.GRIPPER_CLOSED) <= const.GRIPPER_POSITION_TOL
    print("PASS: Gripper close")

    print("\nTEST 8: Gripper open")
    command_gripper(const.GRIPPER_OPEN)
    assert abs(get_gripper_angle() - const.GRIPPER_OPEN) <= const.GRIPPER_POSITION_TOL
    print("PASS: Gripper open")

    print("\n==========================================")
    print("ALL CONTROLLER TESTS PASSED")
    print("==========================================")


# ==========================================
# MAIN - CONTROLLER TEST
# ==========================================

if __name__ == "__main__":
    test_config()
