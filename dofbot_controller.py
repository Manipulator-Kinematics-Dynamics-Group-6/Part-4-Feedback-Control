# ==========================================
# DOFBOT ROBOT CONTROLLER
# ==========================================
#
# Description:
#     Handles trajectory execution, joint commands,
#     gripper control, and robot state retrieval.
#
#     Supports both simulated and physical operation
#     through the TEST flag in constants.py.
#
#     Also manages trajectory timing and simulation
#     pause/resume functionality.
#
# ==========================================

# only needed if running with IPYNB files on Jupyter, comment out otherwise
# import import_ipynb
import numpy as np
import time
import dofbot_kinematics as kinematics
import constants as const
import logging

# ==========================================
# SETUP LOGGING
# ==========================================
logger = logging.getLogger("dofbot")

# ==========================================
# SIMULATED ROBOT STATE
# ==========================================
simulated_q = np.array(
    [90, 90, 90, 90, 90],
    dtype=float
)
simulated_gripper = const.GRIPPER_OPEN

# ==========================================
# HARDWARE CONNECTION
# ==========================================
if not const.TEST:
    from Arm_Lib import Arm_Device
    arm = Arm_Device()

# ==========================================
# JOINT CONTROL
# ==========================================
# Waits until all five joints reach their desired positions.
def wait_for_joints(target_angles, visualizer=None):
    target_angles = np.asarray(
        target_angles,
        dtype=float
    ).flatten()
    if target_angles.size != 5:
        raise ValueError(
            "Target must contain exactly 5 joint angles."
        )
    start_time = time.monotonic()
    while True:
        # ==================================
        # READ CURRENT JOINT POSITIONS
        # ==================================
        current_angles = np.asarray(
            get_joint_angles(),
            dtype=float
        )
        if current_angles.shape != (5,):
            raise RuntimeError(
                "Invalid joint feedback received."
            )
        if not np.all(np.isfinite(current_angles)):
            raise RuntimeError(
                "Joint feedback contains invalid values."
            )
        # ==================================
        # CALCULATE POSITION ERROR
        # ==================================
        error = np.abs(
            target_angles - current_angles
        )
        # Check all five joints
        reached = np.all(
            error <= const.JOINT_POSITION_TOL
        )
        if reached:
            logger.debug(
                "All joints reached target positions. Actual: %s | Errors: %s",
                current_angles.tolist(),
                error.tolist()
            )
            return True
        # ==================================
        # CHECK TIMEOUT
        # ==================================
        elapsed = (
            time.monotonic() - start_time
        )
        if elapsed >= const.JOINT_POSITION_TIMEOUT:
            logger.error(
                "Joint position timeout! Target: %s | Actual: %s | Errors: %s",
                target_angles.tolist(),
                current_angles.tolist(),
                error.tolist()
            )
            return False
        # ==================================
        # PROCESS VISUALIZATION EVENTS
        # ==================================
        if visualizer is not None:
            visualizer.process_events()
        time.sleep(
            const.JOINT_FEEDBACK_INTERVAL
        )

# Sends desired joint angles to the simulated or physical robot.
def command_robot(joint_angles, gripper_angle, movement_time=2500):
    global simulated_q, simulated_gripper
    joint_angles = np.asarray(
        joint_angles,
        dtype=float
    ).flatten()
    if joint_angles.size != 5:
        raise ValueError(
            "Joint angles must contain exactly 5 values."
        )
    logger.debug(
        "Joint command: joints_deg=%s gripper_deg=%s duration_ms=%s",
        joint_angles.tolist(), gripper_angle, movement_time
    )
    if const.TEST:
        # Update simulated joint positions
        simulated_q = joint_angles.copy()
        # Update simulated gripper position
        simulated_gripper = float(gripper_angle)
    else:
        # Send joint angles to physical DOFBOT
        arm.Arm_serial_servo_write6(
            *joint_angles,
            gripper_angle,
            movement_time
        )

# Reads the current joint angles from the simulation or hardware.
def get_joint_angles():
    if const.TEST:
        return simulated_q.copy()
    else:
        time.sleep(.01)
        angles = []
        for joint in range(1, 6):
            angle = arm.Arm_serial_servo_read(
                joint
            )
            if angle is None:
                raise RuntimeError(
                    f"Could not read servo {joint}."
                )
#                 print("Could not read servo " + joint + ".")
            angles.append(angle)
        return np.array(
            angles,
            dtype=float
        )

# ==========================================
# GRIPPER CONTROL
# ==========================================
# Waits until the gripper reaches its target or times out.
def wait_for_gripper(target_angle, visualizer=None):
    start_time = time.monotonic()
    while True:
        current_angle = get_gripper_angle()
        error = abs(target_angle - current_angle)
        if error <= const.GRIPPER_POSITION_TOL:
            logger.debug(
                "Gripper reached target: %.1f deg (error: %.1f deg)",
                current_angle,
                error
            )
            return True
        if time.monotonic() - start_time >= const.GRIPPER_POSITION_TIMEOUT:
            logger.error(
                "Gripper timeout! Target: %.1f deg | Actual: %.1f deg | Error: %.1f deg",
                target_angle,
                current_angle,
                error
            )
            return False
        if visualizer is not None:
            visualizer.process_events()
        time.sleep(const.GRIPPER_FEEDBACK_INTERVAL)

# Commands the gripper and waits for position feedback.
def command_gripper(angle, movement_time=1000, visualizer=None):
    global simulated_gripper
    if not 0 <= angle <= 180:
        raise ValueError(
            "Gripper angle must be between 0 and 180."
        )
    if const.TEST:
        simulated_gripper = float(angle)
    else:
        arm.Arm_serial_servo_write(
            const.GRIPPER_ID,
            angle,
            movement_time
        )
    # Wait for the target position
    reached = wait_for_gripper(
        angle,
        visualizer
    )
    if not reached:
        raise TimeoutError(
            "Gripper failed to reach its target."
        )

# Returns the current gripper servo angle.
def get_gripper_angle():
    if const.TEST:
        return simulated_gripper
    else:
        time.sleep(.01)
        angle = arm.Arm_serial_servo_read(
            const.GRIPPER_ID
        )
        if angle is None:
            raise RuntimeError(
                "Could not read gripper position."
            )
        return angle


# ==========================================
# PID FEEDBACK CONTROL
# ==========================================

# Stores the accumulated error and previous measurement.
pid_integral = np.zeros(5, dtype=float)
pid_previous_actual = None

# Initializes or resets the PID controller state.
def reset_pid():
    global pid_integral, pid_previous_actual

    pid_integral = np.zeros(5, dtype=float)
    pid_previous_actual = None

# Calculates the PID position correction for all five joints.
def calculate_pid(desired_q, actual_q, dt):
    global pid_integral, pid_previous_actual

    # ==================================
    # VALIDATE INPUTS
    # ==================================

    desired_q = np.asarray(desired_q, dtype=float).flatten()
    actual_q = np.asarray(actual_q, dtype=float).flatten()

    if desired_q.size != 5 or actual_q.size != 5:
        raise ValueError("PID requires exactly 5 joint angles.")

    if not np.all(np.isfinite(desired_q)):
        raise ValueError("Desired joint angles contain invalid values.")

    if not np.all(np.isfinite(actual_q)):
        raise ValueError("Actual joint angles contain invalid values.")

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("PID time step must be positive.")

    # ==================================
    # CALCULATE POSITION ERROR
    # ==================================

    error = desired_q - actual_q

    # ==================================
    # PROPORTIONAL TERM
    # ==================================

    P = const.KP * error

    # ==================================
    # INTEGRAL TERM
    # ==================================

    candidate_integral = np.clip(
        pid_integral + error * dt,
        -const.PID_INTEGRAL_LIMIT,
        const.PID_INTEGRAL_LIMIT
    )

    # ==================================
    # DERIVATIVE TERM
    # ==================================

    # Use derivative on measurement to avoid
    # large spikes when desired angles change.
    if pid_previous_actual is None:
        derivative = np.zeros(5, dtype=float)
    else:
        derivative = -(actual_q - pid_previous_actual) / dt

    D = const.KD * derivative

    # ==================================
    # INTEGRAL ANTI-WINDUP
    # ==================================

    proposed = P + const.KI * candidate_integral + D

    winding_up = (
        (proposed > const.PID_MAX_CORRECTION) & (error > 0)
    ) | (
        (proposed < -const.PID_MAX_CORRECTION) & (error < 0)
    )

    pid_integral = np.where(
        winding_up,
        pid_integral,
        candidate_integral
    )

    I = const.KI * pid_integral

    # ==================================
    # LIMIT PID CORRECTION
    # ==================================

    correction = np.clip(
        P + I + D,
        -const.PID_MAX_CORRECTION,
        const.PID_MAX_CORRECTION
    )

    # ==================================
    # CALCULATE COMMANDED POSITION
    # ==================================

    command_q = np.clip(
        desired_q + correction,
        const.JOINT_MIN,
        const.JOINT_MAX
    )

    # Save measurement for the next update.
    pid_previous_actual = actual_q.copy()

    logger.debug(
        "PID | Error: %s | Correction: %s | Command: %s",
        error.tolist(),
        correction.tolist(),
        command_q.tolist()
    )

    return command_q, error, correction



# ==========================================
# EXECUTE TRAJECTORY
# ==========================================

# Executes a trajectory with optional PID feedback control.
# PID and tracking share the same joint feedback readings.
def execute_trajectory(gripper_state, visualizer=None, tracking=None):

    # ==================================
    # TRAJECTORY SETTINGS
    # ==================================

    dt = const.TOTAL_TIME / (const.N - 1)
    movement_time = max(100, int(dt * 1000))

    feedback_interval = const.PID_UPDATE_INTERVAL

    if feedback_interval < 1:
        raise ValueError("PID_UPDATE_INTERVAL must be >= 1.")

    # Tracking uses the same feedback update rate.
    if tracking is not None:
        if const.TRACKING_INTERVAL != feedback_interval:
            raise ValueError(
                "TRACKING_INTERVAL and PID_UPDATE_INTERVAL "
                "must match to share feedback."
            )

    # ==================================
    # INITIALIZE PID
    # ==================================

    reset_pid()

    correction_q = np.zeros(5, dtype=float)

    start_time = time.monotonic()
    last_feedback_time = None
    paused_duration = 0.0

    logger.debug(
        "Trajectory started | N=%d | Duration=%.2fs | PID=%s",
        const.N,
        const.TOTAL_TIME,
        const.PID_ENABLED
    )

    try:
        
        for ii in range(const.N):

            # ==================================
            # PROCESS VISUALIZER EVENTS
            # ==================================

            if visualizer is not None:
                visualizer.process_events()

                paused_duration += (
                    visualizer.wait_if_paused()
                )

            # ==================================
            # MAINTAIN TRAJECTORY TIMING
            # ==================================

            target_time = (
                start_time
                + ii * dt
                + paused_duration
            )

            while True:
                if visualizer is not None:
                    visualizer.process_events()

                    pause_time = (
                        visualizer.wait_if_paused()
                    )

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

            # ==================================
            # SHARED FEEDBACK UPDATE
            # ==================================

            feedback_due = (
                ii % feedback_interval == 0
                or ii == const.N - 1
            )

            if feedback_due and (
                const.PID_ENABLED or tracking is not None
            ):

                # Read actual joint positions once.
                actual_q = get_joint_angles()
                feedback_time = time.monotonic()
                
                # ==================================
                # ALIGN DESIRED POSITION WITH FEEDBACK
                # ==================================

                # Elapsed trajectory time, excluding pauses.
                elapsed = (
                    feedback_time
                    - start_time
                    - paused_duration
                )

                # Convert elapsed time into trajectory progress.
                progress = np.clip(
                    elapsed / const.TOTAL_TIME,
                    0.0,
                    1.0
                )

                # Interpolate desired angles at measurement time.
                desired_feedback_q = np.array([
                    np.interp(
                        progress,
                        const.lam,
                        kinematics.q[joint, :]
                    )
                    for joint in range(5)
                ])

                # Calculate actual feedback interval.
                if last_feedback_time is None:
                    dt_feedback = dt
                else:
                    dt_feedback = (
                        feedback_time - last_feedback_time
                    )

                last_feedback_time = feedback_time

                # ==================================
                # PID CORRECTION
                # ==================================

                if const.PID_ENABLED:
                    command_q, error_q, correction_q = calculate_pid(
                        desired_feedback_q,
                        actual_q,
                        dt_feedback
                    )

                    # Preserve the effective correction
                    # after applying joint limits.
                    correction_q = command_q - desired_feedback_q

                    logger.debug(
                        "PID Frame %d | Error=%s | Correction=%s",
                        ii,
                        error_q.tolist(),
                        correction_q.tolist()
                    )

                # ==================================
                # RECORD TRACKING DATA
                # ==================================

                if tracking is not None:
                    tracking.record(
                        desired_feedback_q,
                        actual_q,
                        measurement_time=feedback_time
                    )

            # ==================================
            # CALCULATE COMMANDED POSITION
            # ==================================

            if const.PID_ENABLED:
                command_q = np.clip(
                    desired_q + correction_q,
                    const.JOINT_MIN,
                    const.JOINT_MAX
                )

            else:
                command_q = desired_q.copy()

            # ==================================
            # COMMAND ROBOT
            # ==================================

            command_robot(
                command_q,
                gripper_state,
                movement_time
            )

            # ==================================
            # UPDATE VISUALIZATION
            # ==================================

            if visualizer is not None:
                visualizer.update(ii)

            # ==================================
            # LOG FRAME DATA
            # ==================================

            logger.debug(
                "Frame %d/%d | Time=%.2fs | Desired=%s | Command=%s",
                ii + 1,
                const.N,
                ii * dt,
                desired_q.tolist(),
                command_q.tolist()
            )

        # ==================================
        # VERIFY FINAL JOINT POSITIONS
        # ==================================

        target_angles = kinematics.q[:, -1].copy()

        logger.debug(
            "Trajectory commands sent. Verifying final position."
        )

        reached = wait_for_joints(
            target_angles,
            visualizer
        )

        if not reached:
            logger.error(
                "Trajectory failed final joint verification."
            )

            raise TimeoutError(
                "Robot failed to reach the final "
                "joint configuration."
            )

        logger.debug(
            "Trajectory completed successfully | "
            "Elapsed=%.3fs",
            time.monotonic() - start_time
        )

    except KeyboardInterrupt:
        logger.warning("Trajectory interrupted by user.")
        raise

    except Exception:
        logger.exception("Trajectory execution failed.")
        raise

    finally:
        logger.debug("Controller trajectory execution finished.")


# ==========================================
# TEST CONFIGURATIONS
# ==========================================
def test_config() :
    qstart = np.array(
        [90, 90, 90, 90, 90],
        dtype=float
    )
    qend = np.array(
        [45, 120, 140, 100, 95],
        dtype=float
    )
    print("\n==========================================")
    print("DOFBOT CONTROLLER TEST")
    print("==========================================")
    # Prevent accidental hardware movement
    assert const.TEST, (
        "Controller tests require TEST = True."
    )
    # ==========================================
    # RESET SIMULATED ROBOT STATE
    # ==========================================
    const.simulated_q = qstart.copy()
    const.simulated_gripper = const.GRIPPER_OPEN
    # ==========================================
    # TEST 1 - INITIAL ROBOT STATE
    # ==========================================
    print("\nTEST 1: Initial robot state")
    actual_q = get_joint_angles()
    actual_gripper = get_gripper_angle()
    assert np.allclose(
        actual_q,
        qstart
    ), (
        f"Initial joint positions incorrect: {actual_q}"
    )
    assert np.isclose(
        actual_gripper,
        const.GRIPPER_OPEN
    ), (
        f"Initial gripper angle incorrect: {actual_gripper}"
    )
    print("PASS: Initial robot state")
    # ==========================================
    # TEST 2 - JOINT COMMAND
    # ==========================================
    print("\nTEST 2: Joint command")
    command_robot(
        qend,
        const.GRIPPER_OPEN
    )
    actual_q = get_joint_angles()
    assert np.allclose(
        actual_q,
        qend
    ), (
        f"Joint command failed. "
        f"Expected: {qend}, Actual: {actual_q}"
    )
    print("PASS: Joint command")
    # ==========================================
    # TEST 3 - JOINT POSITION FEEDBACK
    # ==========================================
    print("\nTEST 3: Joint position feedback")
    reached = wait_for_joints(
        qend
    )
    assert reached, (
        "Joint position verification failed."
    )
    print("PASS: Joint position feedback")
    # ==========================================
    # TEST 4 - TRAJECTORY GENERATION
    # ==========================================
    print("\nTEST 4: Trajectory generation")
    kinematics.generate_trajectory(
        qstart,
        qend
    )
    assert kinematics.q.shape == (
        5,
        kinematics.N
    ), (
        "Generated trajectory has incorrect dimensions."
    )
    assert np.allclose(
        kinematics.q[:, 0],
        qstart
    ), (
        "Trajectory does not start at qstart."
    )
    assert np.allclose(
        kinematics.q[:, -1],
        qend
    ), (
        "Trajectory does not end at qend."
    )
    assert np.all(
        np.isfinite(kinematics.q)
    ), (
        "Trajectory contains invalid numerical values."
    )
    print("PASS: Trajectory generation")
    # ==========================================
    # TEST 5 - TRAJECTORY EXECUTION
    # ==========================================
    print("\nTEST 5: Trajectory execution")
    # Reset simulated robot to initial configuration
    const.simulated_q = qstart.copy()
    execute_trajectory(
        const.GRIPPER_OPEN
    )
    actual_q = get_joint_angles()
    joint_error = np.abs(
        qend - actual_q
    )
    assert np.all(
        joint_error <= const.JOINT_POSITION_TOL
    ), (
        f"Trajectory execution failed. "
        f"Joint errors: {joint_error}"
    )
    print("PASS: Trajectory execution")
    # ==========================================
    # TEST 6 - FORWARD KINEMATICS
    # ==========================================
    print("\nTEST 6: Forward kinematics")
    Rot, Pot = kinematics.fwdkin_Dofbot(
        actual_q
    )
    assert Rot.shape == (3, 3), (
        "Rotation matrix has incorrect dimensions."
    )
    assert Pot.shape == (3,), (
        "End-effector position has incorrect dimensions."
    )
    assert np.allclose(
        Rot.T @ Rot,
        np.eye(3),
        atol=1e-6
    ), (
        "Rotation matrix is not orthogonal."
    )
    assert np.isclose(
        np.linalg.det(Rot),
        1.0,
        atol=1e-6
    ), (
        "Rotation matrix has invalid determinant."
    )
    assert np.all(
        np.isfinite(Pot)
    ), (
        "End-effector position contains invalid values."
    )
    print("PASS: Forward kinematics")
    # ==========================================
    # TEST 7 - GRIPPER CLOSE
    # ==========================================
    print("\nTEST 7: Gripper close")
    command_gripper(
        const.GRIPPER_CLOSED
    )
    actual_gripper = get_gripper_angle()
    assert abs(
        const.GRIPPER_CLOSED - actual_gripper
    ) <= const.GRIPPER_POSITION_TOL, (
        f"Gripper failed to close. "
        f"Expected: {const.GRIPPER_CLOSED}, "
        f"Actual: {actual_gripper}"
    )
    print("PASS: Gripper close")
    # ==========================================
    # TEST 8 - GRIPPER OPEN
    # ==========================================
    print("\nTEST 8: Gripper open")
    command_gripper(
        const.GRIPPER_OPEN
    )
    actual_gripper = get_gripper_angle()
    assert abs(
        const.GRIPPER_OPEN - actual_gripper
    ) <= const.GRIPPER_POSITION_TOL, (
        f"Gripper failed to open. "
        f"Expected: {const.GRIPPER_OPEN}, "
        f"Actual: {actual_gripper}"
    )
    print("PASS: Gripper open")
    # ==========================================
    # TEST COMPLETE
    # ==========================================
    print("\n==========================================")
    print("ALL CONTROLLER TESTS PASSED")
    print("==========================================")

# ==========================================
# MAIN - CONTROLLER TEST
# ==========================================
if __name__ == "__main__":
    test_config()
