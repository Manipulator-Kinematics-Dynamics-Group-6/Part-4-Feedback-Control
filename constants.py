"""DOFBOT configuration: edit settings here, not in the control modules."""

import numpy as np


# ==========================================
# QUICK SETTINGS - EDIT MOST OFTEN
# ==========================================

# True = simulation only; False = command the physical DOFBOT.
TEST = True

# Duration of each MOVE_TO trajectory (seconds).
TOTAL_TIME = 3

# Select one of the command sequences defined below.
# To add a sequence, define it in section 2 and select it here.
# ACTIVE_SEQUENCE is assigned after the sequences are defined.


# ==========================================
# COMMAND SEQUENCES - EDIT AS NEEDED
# ==========================================

# Each entry is (command, target joint angles).
# MOVE_TO targets contain five angles in degrees.
# GRAB / DROP do not require target joint angles.
PICK_AND_PLACE = [
    ("MOVE_TO", [45, 120, 140, 100, 95]),
    ("GRAB", None),
    ("MOVE_TO", [90, 90, 90, 90, 90]),
    ("DROP", None),
]

TEST_SEQUENCE = [
    ("MOVE_TO", [45, 120, 140, 100, 95]),
    ("MOVE_TO", [90, 90, 90, 90, 90]),
]

RESET = [
    ("MOVE_TO", [90, 90, 90, 90, 90]),
    ("DROP", None),
]

RESET_OTHER = [
    ("MOVE_TO", [45, 120, 140, 100, 95]),
    ("GRAB", None),
]

# Choose the sequence to run (change only this line).
ACTIVE_SEQUENCE = RESET


# ==========================================
# SERVO SETTINGS
# ==========================================

# Gripper servo number and target angles (degrees).
GRIPPER_ID = 6
GRIPPER_OPEN = 60
GRIPPER_CLOSED = 130
GRIPPER_WAIT = 1.5  # Seconds; retained for existing callers.

# Joint limits in degrees. Verify against the actual robot before hardware use.
JOINT_MIN = np.array([0, 0, 0, 0, 0], dtype=float)
JOINT_MAX = np.array([180, 180, 180, 180, 180], dtype=float)



# ==========================================
# FEEDBACK AND POSITION VERIFICATION
# ==========================================

# Record actual joint feedback once every this many trajectory frames.
TRACKING_INTERVAL = 5

# Joint target verification.
JOINT_POSITION_TOL = 3.0       # Degrees
JOINT_POSITION_TIMEOUT = 5.0   # Seconds
JOINT_FEEDBACK_INTERVAL = 0.05 # Seconds between feedback reads

# Gripper target verification.
GRIPPER_POSITION_TOL = 3.0       # Degrees
GRIPPER_POSITION_TIMEOUT = 5.0   # Seconds
GRIPPER_FEEDBACK_INTERVAL = 0.05 # Seconds between feedback reads


# ==========================================
# TRAJECTORY AND ROBOT GEOMETRY
# ==========================================

# Number of points per MOVE_TO trajectory.
N = 100
lam = np.linspace(0, 1, N)

# Measured/assumed robot link lengths (meters).
# Normally keep these fixed unless recalibrating the robot model.
l0 = 0.061
l1 = 0.0435
l2 = 0.08285
l3 = 0.08285
l4 = 0.07385
l5 = 0.05457

LINK_LENGTHS = [l0, l1, l2, l3, l4, l5]


# ==========================================
# INVERSE KINEMATICS SETTINGS
# ==========================================

POSITION_TOL = 1e-4      # Meters
ORIENTATION_TOL = 1e-3   # Radians
IK_ATTEMPTS = 24         # Number of solver attempts


# ==========================================
# LOGGING AND REPORT FILES
# ==========================================

# INFO: major events in console; DEBUG: detailed events in log file.
CONSOLE_LOG_LEVEL = "INFO"
FILE_LOG_LEVEL = "DEBUG"
LOG_FOLDER = "reports"
