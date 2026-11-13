# ==========================================
# DOFBOT MAIN PROGRAM
# ==========================================
# Coordinates command execution, trajectory tracking,
# visualization, and experiment logging.

# only needed if running with IPYNB files on Jupyter, comment out otherwise 
# import import_ipynb 

import time
import json
import logging

from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import constants as const
import dofbot_kinematics as kinematics
import dofbot_controller as controller
import dofbot_tracking as tracking_data


# ==========================================
# EXPERIMENT LOGGING
# ==========================================

# Formats diagnostic events as one JSON object per line.
class JSONFormatter(logging.Formatter):

    def format(self, record):
        entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage()
        }
        if record.exc_info:
            entry["traceback"] = self.formatException(record.exc_info)
        return json.dumps(entry)


# Configures concise console messages and detailed JSONL events.
def setup_logging(run_folder):
    filepath = Path(run_folder) / "events.jsonl"

    logger = logging.getLogger("dofbot")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.run_folder = run_folder

    # Prevent duplicate messages on repeated JupyterLab runs.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler()
    console.setLevel(getattr(logging, const.CONSOLE_LOG_LEVEL.upper()))
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console)

    file_handler = logging.FileHandler(filepath, encoding="utf-8")
    file_handler.setLevel(getattr(logging, const.FILE_LOG_LEVEL.upper()))
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    logger.debug("Detailed logging enabled: %s", filepath)
    return logger


# ==========================================
# EXECUTE COMMAND SEQUENCE
# ==========================================

# Runs commands while recording tracking data and detailed events.
def run_commands(visualizer=None):
    
    # ==========================================
    # EXPERIMENT DIRECTORY
    # ==========================================

    run_id = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S_%f"
    )

    sequence_name = next(
        (
            name for name in (
                "PICK_AND_PLACE",
                "TEST_SEQUENCE",
                "RESET",
                "RESET_OTHER"
            )
            if getattr(const, name) is const.ACTIVE_SEQUENCE
        ),
        "CUSTOM"
    )

    mode = "SIM" if const.TEST else "HW"

    run_folder = Path(const.LOG_FOLDER) / (
        f"{run_id}_{sequence_name}_{mode}"
    )

    run_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # Initialize logging when called from a notebook or visualizer.
    # A new report folder is created for every run, including notebook calls.
    logger = setup_logging(run_folder)

    sequence = const.ACTIVE_SEQUENCE
    total_commands = len(sequence)
    total_moves = sum(1 for command_type, _ in sequence
                      if command_type == "MOVE_TO")
    move_number = 0
    tracking = tracking_data.TrajectoryRecorder()

    logger.info("DOFBOT sequence started | Mode: %s | Commands: %d",
                "SIMULATION" if const.TEST else "HARDWARE", total_commands)

    # Read initial state; detailed data goes to JSONL only.
    gripper_state = controller.get_gripper_angle()
    joints_state = np.asarray(controller.get_joint_angles(), dtype=float)
    logger.debug("Initial joints (deg): %s", joints_state.tolist())
    logger.debug("Initial gripper (deg): %.2f", gripper_state)

    if visualizer is not None:
        visualizer.begin_sequence(total_commands)

    completed = False
    try:
        for command_number, (command_type, value) in enumerate(sequence, start=1):

            # ==================================
            # VALIDATE COMMAND
            # ==================================
            if command_type not in ("MOVE_TO", "GRAB", "DROP"):
                logger.warning("[%d/%d] Unknown command: %s (skipped)",
                               command_number, total_commands, command_type)
                continue

            qstart = np.asarray(controller.get_joint_angles(), dtype=float)
            logger.info("[%d/%d] %s started",
                        command_number, total_commands, command_type)
            logger.debug("[%d/%d] Initial joints (deg): %s",
                         command_number, total_commands, qstart.tolist())
            command_started = time.monotonic()

            # ==================================
            # MOVE ARM
            # ==================================
            if command_type == "MOVE_TO":
                move_number += 1
                qend = np.asarray(value, dtype=float).flatten()
                if qend.size != 5:
                    raise ValueError("MOVE_TO requires 5 joint angles.")

                logger.debug("[%d/%d] Desired joints (deg): %s",
                             command_number, total_commands, qend.tolist())
                kinematics.generate_trajectory(qstart, qend)

                if visualizer is not None:
                    visualizer.new_command(
                        command_type, gripper_state,
                        command_number=command_number,
                        command_total=total_commands,
                        move_number=move_number,
                        move_total=total_moves
                    )

                tracking.begin_command(move_number)
                controller.execute_trajectory(
                    gripper_state, visualizer, tracking=tracking
                )

            # ==================================
            # GRIPPER COMMANDS
            # ==================================
            else:
                if visualizer is not None:
                    visualizer.record_gripper_command(
                        command_type, gripper_state, qstart,
                        command_number=command_number,
                        command_total=total_commands,
                        move_total=total_moves
                    )

                target_gripper = (const.GRIPPER_CLOSED if command_type == "GRAB"
                                  else const.GRIPPER_OPEN)
                logger.debug("[%d/%d] Gripper target (deg): %.2f",
                             command_number, total_commands, target_gripper)
                controller.command_gripper(target_gripper, visualizer=visualizer)

            # ==================================
            # UPDATE ROBOT STATE
            # ==================================
            joints_state = np.asarray(controller.get_joint_angles(), dtype=float)
            gripper_state = controller.get_gripper_angle()
            logger.debug("[%d/%d] Final joints (deg): %s",
                         command_number, total_commands, joints_state.tolist())
            logger.debug("[%d/%d] Final gripper (deg): %.2f",
                         command_number, total_commands, gripper_state)

            if visualizer is not None:
                visualizer.set_gripper_angle(gripper_state)

            logger.info("[%d/%d] %s completed - %.2f s",
                        command_number, total_commands, command_type,
                        time.monotonic() - command_started)

        completed = True

    except KeyboardInterrupt:
        logger.warning("Command sequence interrupted by user")
        raise
    except Exception:
        logger.exception("Command sequence failed")
        raise
    finally:
        if visualizer is not None:
            try:
                visualizer.end_sequence()
            except Exception:
                logger.exception("Unable to end visualization sequence")

        # Preserve partial experiment data after an interrupted run.
        for name, save_method in (("CSV", tracking.save_csv),
                                  ("JSON summary", tracking.save_report)):
            try:
                path = save_method(folder=str(logger.run_folder))
                if path is not None:
                    logger.info("%s saved: %s", name, path)
            except Exception:
                logger.exception("Failed to save %s", name)

        # Avoid showing plots in a failed run; data is still saved above.
        if completed:
            try:
                if tracking.samples:
                    tracking.plot()
                    plt.show()
            except Exception:
                logger.exception("Unable to display tracking graphs")

    logger.info("Command sequence finished successfully")
    logger.debug("Final joint positions (deg): %s", joints_state.tolist())
    logger.debug("Final gripper angle (deg): %.2f", gripper_state)
    return joints_state


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    run_commands()
