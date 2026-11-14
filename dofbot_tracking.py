# ==========================================
# DOFBOT TRAJECTORY TRACKING
# ==========================================
#
# Description:
#     Records and analyzes the DOFBOT's trajectory execution performance.
#
#     Collects desired and actual joint angles, the corresponding
#     end-effector positions and orientations.
#
#     Computes tracking errors and performance metrics, including RMSE, 
#     maximum error, and final position error.
#
#     Exports experiment measurements to CSV and performance summaries to JSON.
#
#     Provides interactive graphs for comparing desired and actual trajectories.
#
# ==========================================

# only needed if running with IPYNB files on Jupyter, comment out otherwise 
# import import_ipynb 

import time
import warnings
import csv
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from pathlib import Path
from datetime import datetime
from matplotlib.widgets import Button

import dofbot_kinematics as kinematics


# ==========================================
# TRAJECTORY TRACKING
# ==========================================

class TrajectoryRecorder:

    # Initializes the desired and actual trajectory history.
    def __init__(self):

        self.start_time = time.monotonic()

        self.samples = []
        self.command_boundaries = []


    # Records the start of a new MOVE_TO command.
    def begin_command(self, command_number):

        elapsed = time.monotonic() - self.start_time

        self.command_boundaries.append(
            (elapsed, command_number)
        )


    # Records joint angles and calculated end-effector poses.
    def record(self, desired_q, actual_q, measurement_time=None):

        if measurement_time is None:
            measurement_time = time.monotonic()
        
        desired_q = np.asarray(
            desired_q, dtype=float
        ).reshape(5)

        actual_q = np.asarray(
            actual_q, dtype=float
        ).reshape(5)

        if not (
            np.all(np.isfinite(desired_q))
            and np.all(np.isfinite(actual_q))
        ):
            raise ValueError(
                "Invalid joint feedback received."
            )

        Rot_desired, Pot_desired = (
            kinematics.fwdkin_Dofbot(desired_q)
        )

        Rot_actual, Pot_actual = (
            kinematics.fwdkin_Dofbot(actual_q)
        )

        with warnings.catch_warnings():

            warnings.filterwarnings(
                "ignore",
                message="Gimbal lock detected"
            )

            rpy_desired = Rotation.from_matrix(
                Rot_desired
            ).as_euler("ZYX", degrees=True)[::-1]

            rpy_actual = Rotation.from_matrix(
                Rot_actual
            ).as_euler("ZYX", degrees=True)[::-1]

        orientation_error = np.rad2deg(
            Rotation.from_matrix(
                Rot_desired @ Rot_actual.T
            ).magnitude()
        )

        self.samples.append({
            "time": measurement_time - self.start_time,
            "desired_q": desired_q.copy(),
            "actual_q": actual_q.copy(),
            "desired_p": Pot_desired.copy(),
            "actual_p": Pot_actual.copy(),
            "desired_rpy": rpy_desired.copy(),
            "actual_rpy": rpy_actual.copy(),
            "orientation_error": orientation_error
        })


    # Creates comparison graphs for each vector component.
    def plot_components(
        self, t, desired, actual,
        labels, title, ylabel
    ):

        fig, axes = plt.subplots(
            len(labels),
            1,
            figsize=(11, 2 * len(labels) + 1),
            sharex=True
        )

        fig.suptitle(title)

        for ii, label in enumerate(labels):

            axes[ii].plot(
                t,
                desired[:, ii],
                "--",
                label=f"{label} desired"
            )

            axes[ii].plot(
                t,
                actual[:, ii],
                "-",
                label=f"{label} actual"
            )

            axes[ii].set_ylabel(ylabel)
            axes[ii].legend(loc="best")
            axes[ii].grid(True, alpha=0.3)

            self.mark_commands(axes[ii])

        axes[-1].set_xlabel("Elapsed time (s)")

        fig.tight_layout()


    # Marks the beginning of each MOVE_TO command.
    def mark_commands(self, ax):

        for elapsed, number in self.command_boundaries:

            ax.axvline(
                elapsed,
                color="gray",
                linewidth=0.8,
                alpha=0.5
            )


    
    # ==========================================
    # TRACKING GRAPHS
    # ==========================================

    # Displays desired and actual tracking in one tabbed window.
    def plot(self):

        if not self.samples:
            print("No tracking data to plot.")
            return

        # ==================================
        # PREPARE TRACKING DATA
        # ==================================

        t = np.array([s["time"] for s in self.samples])

        qd = np.array([s["desired_q"] for s in self.samples])
        qa = np.array([s["actual_q"] for s in self.samples])

        pd = np.array([s["desired_p"] for s in self.samples]) * 1000
        pa = np.array([s["actual_p"] for s in self.samples]) * 1000

        rd = np.array([s["desired_rpy"] for s in self.samples])
        ra = np.array([s["actual_rpy"] for s in self.samples])

        rd = np.rad2deg(np.unwrap(np.deg2rad(rd), axis=0))
        ra = np.rad2deg(np.unwrap(np.deg2rad(ra), axis=0))

        joint_error = np.max(np.abs(qd - qa), axis=1)
        position_error = np.linalg.norm(pd - pa, axis=1)

        orientation_error = np.array([
            s["orientation_error"] for s in self.samples
        ])

        # ==================================
        # CONFIGURE GRAPH TABS
        # ==================================

        tabs = [
            (
                "Joints", qd, qa,
                ["J1", "J2", "J3", "J4", "J5"],
                "Angle (deg)"
            ),
            (
                "EE Position", pd, pa,
                ["X", "Y", "Z"],
                "Position (mm)"
            ),
            (
                "EE Orientation", rd, ra,
                ["Roll", "Pitch", "Yaw"],
                "Angle (deg)"
            ),
            (
                "Errors",
                np.column_stack((
                    joint_error,
                    position_error,
                    orientation_error
                )),
                None,
                [
                    "Max Joint Error (deg)",
                    "EE Position Error (mm)",
                    "EE Orientation Error (deg)"
                ],
                ""
            )
        ]

        # ==================================
        # CREATE SINGLE WINDOW
        # ==================================

        fig = plt.figure(figsize=(12, 9))

        try:
            fig.canvas.manager.set_window_title(
                "DOFBOT Tracking Analysis"
            )
        except (AttributeError, TypeError):
            pass

        buttons = []
        button_axes = []

        # ==================================
        # DISPLAY SELECTED TAB
        # ==================================

        def show_tab(index):

            # Remove existing graphs, keeping the tab buttons.
            for ax in list(fig.axes):
                if ax not in button_axes:
                    fig.delaxes(ax)

            name, desired, actual, labels, ylabel = tabs[index]

            fig.suptitle(
                f"DOFBOT Tracking - {name}",
                fontsize=14,
                y=0.965
            )

            grid = fig.add_gridspec(
                len(labels), 1,
                left=0.15,
                right=0.96,
                bottom=0.08,
                top=0.87,
                hspace=0.3
            )

            for ii, label in enumerate(labels):

                ax = fig.add_subplot(grid[ii, 0])

                ax.plot(
                    t, desired[:, ii],
                    "--" if actual is not None else "-",
                    label="Desired" if actual is not None else label
                )

                if actual is not None:
                    ax.plot(
                        t, actual[:, ii],
                        label="Actual"
                    )
                    ax.legend(loc="best", fontsize=8)

                ax.set_ylabel(
                    f"{label}\n{ylabel}" if ylabel else label
                )

                ax.grid(True, alpha=0.3)

                # Mark command boundaries.
                for boundary_time, _ in self.command_boundaries:
                    ax.axvline(
                        boundary_time,
                        color="gray",
                        alpha=0.4,
                        linewidth=0.8
                    )

                if ii == len(labels) - 1:
                    ax.set_xlabel("Elapsed time (s)")

            # Highlight selected tab.
            for ii, button in enumerate(buttons):
                button.ax.set_facecolor(
                    "0.75" if ii == index else "0.92"
                )

            fig.canvas.draw_idle()

        # ==================================
        # CREATE TAB BUTTONS
        # ==================================

        for ii, tab in enumerate(tabs):

            ax = fig.add_axes([
                0.035 + ii * 0.245,
                0.895,
                0.23,
                0.045
            ])

            button_axes.append(ax)

            button = Button(
                ax,
                tab[0],
                color="0.92",
                hovercolor="0.82"
            )

            button.on_clicked(
                lambda event, index=ii: show_tab(index)
            )

            buttons.append(button)

        # Keep callbacks alive while the window is open.
        fig._tracking_buttons = buttons

        # Display first tab.
        show_tab(0)

        return fig

    
    # ==========================================
    # EXPORT TRACKING DATA
    # ==========================================

    # Converts recorded samples into rows for CSV export.
    def get_data_rows(self):

        rows = []

        for sample in self.samples:

            desired_q = sample["desired_q"]
            actual_q = sample["actual_q"]

            desired_p = sample["desired_p"]
            actual_p = sample["actual_p"]

            desired_rpy = sample["desired_rpy"]
            actual_rpy = sample["actual_rpy"]

            row = {
                "time_s": float(sample["time"])
            }

            # Identify the active movement.
            command_number = 0

            for boundary_time, number in self.command_boundaries:

                if sample["time"] >= boundary_time:
                    command_number = number

            row["move_number"] = command_number

            # Joint angles in degrees.
            for ii in range(5):

                row[f"J{ii+1}_desired_deg"] = float(
                    desired_q[ii]
                )

                row[f"J{ii+1}_actual_deg"] = float(
                    actual_q[ii]
                )

                row[f"J{ii+1}_error_deg"] = float(
                    desired_q[ii] - actual_q[ii]
                )

            # End-effector position in millimeters.
            for ii, axis in enumerate(["X", "Y", "Z"]):

                row[f"{axis}_desired_mm"] = float(
                    desired_p[ii] * 1000
                )

                row[f"{axis}_actual_mm"] = float(
                    actual_p[ii] * 1000
                )

                row[f"{axis}_error_mm"] = float(
                    (desired_p[ii] - actual_p[ii]) * 1000
                )

            # End-effector orientation in degrees.
            for ii, axis in enumerate(
                ["roll", "pitch", "yaw"]
            ):

                row[f"{axis}_desired_deg"] = float(
                    desired_rpy[ii]
                )

                row[f"{axis}_actual_deg"] = float(
                    actual_rpy[ii]
                )

            # Overall tracking errors.
            row["joint_error_max_deg"] = float(
                np.max(np.abs(desired_q - actual_q))
            )

            row["EE_position_error_mm"] = float(
                np.linalg.norm(
                    desired_p - actual_p
                ) * 1000
            )

            row["EE_orientation_error_deg"] = float(
                sample["orientation_error"]
            )

            rows.append(row)

        return rows


    # Saves recorded trajectory data to a CSV file.
    def save_csv(self, folder="reports"):

        rows = self.get_data_rows()

        if not rows:

            print("No tracking data to export.")
            return None

        Path(folder).mkdir(
            parents=True,
            exist_ok=True
        )

        # Reuse the same identifier for the CSV and report.
        if not hasattr(self, "_report_id"):

            self._report_id = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

        filepath = Path(folder) / (
            f"dofbot_{self._report_id}.csv"
        )

        with open(
            filepath,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(rows[0].keys())
            )

            writer.writeheader()
            writer.writerows(rows)

        print(f"Tracking data saved: {filepath}")

        return filepath


    # ==========================================
    # PERFORMANCE METRICS
    # ==========================================

    # Calculates performance metrics for a group of samples.
    def calculate_metrics(self, samples=None):

        if samples is None:

            samples = self.samples

        if len(samples) == 0:

            return {}

        q_desired = np.array([
            s["desired_q"] for s in samples
        ])

        q_actual = np.array([
            s["actual_q"] for s in samples
        ])

        p_desired = np.array([
            s["desired_p"] for s in samples
        ])

        p_actual = np.array([
            s["actual_p"] for s in samples
        ])

        orientation_error = np.array([
            s["orientation_error"] for s in samples
        ])

        # Joint tracking error.
        joint_error = q_desired - q_actual

        # End-effector position error in mm.
        position_error = (
            p_desired - p_actual
        ) * 1000

        position_error_magnitude = np.linalg.norm(
            position_error,
            axis=1
        )

        metrics = {
            "samples": len(samples),

            "duration_s": float(
                samples[-1]["time"]
                - samples[0]["time"]
            ),

            "joint_rmse_deg": {},

            "joint_max_error_deg": {},

            "EE_position_rmse_mm": float(
                np.sqrt(
                    np.mean(position_error_magnitude ** 2)
                )
            ),

            "EE_position_max_error_mm": float(
                np.max(position_error_magnitude)
            ),

            "EE_position_final_error_mm": float(
                position_error_magnitude[-1]
            ),

            "EE_orientation_rmse_deg": float(
                np.sqrt(
                    np.mean(orientation_error ** 2)
                )
            ),

            "EE_orientation_max_error_deg": float(
                np.max(orientation_error)
            ),

            "EE_orientation_final_error_deg": float(
                orientation_error[-1]
            )
        }

        # Calculate metrics for individual joints.
        for ii in range(5):

            error = joint_error[:, ii]

            metrics["joint_rmse_deg"][f"J{ii+1}"] = float(
                np.sqrt(np.mean(error ** 2))
            )

            metrics["joint_max_error_deg"][f"J{ii+1}"] = float(
                np.max(np.abs(error))
            )

        return metrics


    # ==========================================
    # SAVE PERFORMANCE REPORT
    # ==========================================

    # Saves overall and per-movement performance metrics.
    def save_report(self, folder="reports"):

        if not self.samples:

            print("No tracking data to analyze.")
            return None

        Path(folder).mkdir(
            parents=True,
            exist_ok=True
        )

        if not hasattr(self, "_report_id"):

            self._report_id = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

        report = {
            "report_id": self._report_id,
            "overall": self.calculate_metrics(),
            "movements": {}
        }

        # Group recorded samples by MOVE_TO command.
        for index, (start_time, move_number) in enumerate(
            self.command_boundaries
        ):

            if index + 1 < len(self.command_boundaries):

                end_time = self.command_boundaries[
                    index + 1
                ][0]

            else:

                end_time = float("inf")

            move_samples = [
                sample
                for sample in self.samples
                if start_time <= sample["time"] < end_time
            ]

            if move_samples:

                report["movements"][
                    str(move_number)
                ] = self.calculate_metrics(move_samples)

        filepath = Path(folder) / (
            f"dofbot_{self._report_id}_summary.json"
        )

        with open(
            filepath,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                report,
                file,
                indent=4
            )

        # ==================================
        # PRINT PERFORMANCE SUMMARY
        # ==================================

        metrics = report["overall"]

        print("\n==========================================")
        print("DOFBOT PERFORMANCE REPORT")
        print("==========================================")

        print(f"Samples: {metrics['samples']}")
        print(f"Duration: {metrics['duration_s']:.2f} s")

        print("\nJoint RMSE:")

        for joint, value in metrics["joint_rmse_deg"].items():

            print(f"  {joint}: {value:.3f} deg")

        print("\nEnd-Effector Tracking:")

        print(
            f"  Position RMSE: "
            f"{metrics['EE_position_rmse_mm']:.3f} mm"
        )

        print(
            f"  Max Position Error: "
            f"{metrics['EE_position_max_error_mm']:.3f} mm"
        )

        print(
            f"  Final Position Error: "
            f"{metrics['EE_position_final_error_mm']:.3f} mm"
        )

        print(
            f"  Orientation RMSE: "
            f"{metrics['EE_orientation_rmse_deg']:.3f} deg"
        )

        print(
            f"  Max Orientation Error: "
            f"{metrics['EE_orientation_max_error_deg']:.3f} deg"
        )

        print("\n==========================================")
        print(f"Performance report saved: {filepath}")
        print("==========================================")

        return filepath
    