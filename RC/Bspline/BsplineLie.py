import numpy as np
from collections import OrderedDict
import manifpy as lie
from numpy.linalg import inv
def rotation_angle_2d(R):
    """Extract the rotation angle (in radians) from a 2D rotation matrix."""
    angle = np.arctan2(R[1, 0], R[0, 0])
    return angle

class BsplineLie:
    def __init__(self):
        """Default constructor."""
        self._dt = None  # Uniform sampling time for our control points
        self._timestamp_start = None  # Start time of the system
        self._control_points = OrderedDict()  # Dictionary for aligned Eigen4D matrix (4x4 matrices)
        self._DOF = 2

    def feed_trajectory(self, traj_points : list):
        """
        Will feed in a series of poses that we will then convert into control points.

        Our control points need to be uniformly spaced over the trajectory, thus given a trajectory we will
        uniformly sample based on the average spacing between the pose points specified.

        :param traj_points: List of trajectory poses [[timestamp(s), SE(2 or 3)]].
        """
        # Find the average frequency to use as our uniform timesteps
        sumdt = 0
        for i in range(len(traj_points) - 1):
            sumdt +=traj_points[i + 1][0] -  traj_points[i][0]
        self._dt = sumdt / (len(traj_points) - 1)
        self._dt = max(self._dt, 0.01)  # Ensure dt is at least 0.05
        print(f"[B-SPLINE]: control point dt = {self._dt:.3f} (original dt of {sumdt / (len(traj_points) - 1):.3f})")

        # Convert all our trajectory points to dictionary matrices
        trajectory_points = OrderedDict()
        for point in traj_points:
            trajectory_points[point[0]] = point[1]  # Store with timestamp

        # Get the oldest timestamp
        timestamp_min = min(trajectory_points.keys())
        timestamp_max = max(trajectory_points.keys())
        print(f"[B-SPLINE]: trajectory start time = {timestamp_min:.6f}")
        print(f"[B-SPLINE]: trajectory end time = {timestamp_max:.6f}")
        self._DOF = trajectory_points[timestamp_max].DoF
        # Create spline control points
        timestamp_curr = timestamp_min
        while True:
            # Get bounding poses for the current time
            t0, pose0, t1, pose1 = self.find_bounding_poses(timestamp_curr, trajectory_points)
            if pose0 is None or pose1 is None:  # Check if we found bounding poses
                break
            
            # Linear interpolation and append to our control points
            lambda_ = (timestamp_curr - t0) / (t1 - t0)
            pose_interp = pose0.rplus(lambda_ * pose1.rminus(pose0))
            self._control_points[timestamp_curr] = pose_interp
            timestamp_curr += self._dt

        # # The start time of the system is two dt in since we need at least two older control points
        self.timestamp_start = timestamp_min + 2 * self._dt
        # self.control_points = trajectory_points

        print(f"[B-SPLINE]: start trajectory time of {self.timestamp_start:.6f}")

    def find_bounding_poses(self, timestamp_curr, trajectory_points):
        """
        Find two bounding poses for a given timestamp.

        :param timestamp_curr: Desired timestamp we want to get two bounding poses of.
        :return: A tuple containing t0, pose0, t1, pose1 if found, else (None, None, None, None).
        """
        # Set the default values
        t0, t1 = -1, -1
        pose0, pose1 = None, None  # Identity matrices

        # Find the bounding poses
        found_older = False
        found_newer = False

        # Find the bounding poses for interpolation
        lower_bound = next((t for t in trajectory_points.keys() if t >= timestamp_curr), None)
        upper_bound = next((t for t in trajectory_points.keys() if t > timestamp_curr), None)

        # Get the corresponding poses
        if lower_bound is not None:
            if lower_bound == timestamp_curr:
                found_older = True
            else:
                # If it's not the exact timestamp, look for the previous one
                lower_bound = max(t for t in trajectory_points.keys() if t < timestamp_curr)
                found_older = True

        if upper_bound is not None:
            found_newer = True

        # If we found the older one, set it
        if found_older:
            t0 = lower_bound
            pose0 = trajectory_points[t0]

        # If we found the newest one, set it
        if found_newer:
            t1 = upper_bound
            pose1 = trajectory_points[t1]

        # Assert the timestamps
        if found_older and found_newer:
            assert t0 < t1

        # Return true if we found both bounds
        return (t0, pose0, t1, pose1) if found_older and found_newer else (None, None, None, None)

    def find_bounding_control_points(self, timestamp):
        """
        Find four bounding control points for a given timestamp.

        :param timestamp: Desired timestamp we want to get four bounding poses of.
        :param poses: Ordered dictionary of poses and timestamps.
        :return: True if all four bounding poses are found, else False.
        """
        # Set the default values
        t0, t1, t2, t3 = -1, -1, -1, -1
        pose0, pose1, pose2, pose3 = None, None, None, None

        # Get the two bounding poses
        t1, pose1, t2, pose2 = self.find_bounding_poses(timestamp, self._control_points)

        # Return false if this was a failure
        if t1 == None:
            return False, t0, pose0, t1, pose1, t2, pose2, t3, pose3

        # Now find the poses that are below and above
        iter_t1 = list(self._control_points.keys()).index(t1)
        iter_t2 = list(self._control_points.keys()).index(t2)

        # Check that t1 is not the first timestamp
        if iter_t1 == 0:
            return False, t0, pose0, t1, pose1, t2, pose2, t3, pose3

        # Move the older pose backwards in time
        # Move the newer one forwards in time
        t0 = list(self._control_points.keys())[iter_t1 - 1]
        pose0 = self._control_points[t0]

        # Check that it is valid
        if iter_t2 + 1 >= len(self._control_points):
            return False, t0, pose0, t1, pose1, t2, pose2, t3, pose3

        t3 = list(self._control_points.keys())[iter_t2 + 1]
        pose3 = self._control_points[t3]

        # Assert the timestamps
        assert t0 < t1
        assert t1 < t2
        assert t2 < t3

        # Return true if we found all four bounding poses
        return True, t0, pose0, t1, pose1, t2, pose2, t3, pose3

    def get_pose(self, timestamp):
        """
        Get the pose at a specific timestamp.

        :param timestamp: Desired timestamp for which to get the pose.
        :return: True if successful, False otherwise.
        """
        # Get the bounding poses for the desired timestamp
        t0, t1, t2, t3 = -1, -1, -1, -1
        pose0, pose1, pose2, pose3 = None, None, None, None

        success, t0, pose0, t1, pose1, t2, pose2, t3, pose3 = self.find_bounding_control_points(timestamp)

        # Return failure if we can't get bounding poses
        if not success:

            return False, None

        # Our De Boor-Cox matrix scalars
        DT = (t2 - t1)
        u = (timestamp - t1) / DT
        b0 = 1.0 / 6.0 * (5 + 3 * u - 3 * u ** 2 + u ** 3)
        b1 = 1.0 / 6.0 * (1 + 3 * u + 3 * u ** 2 - 2 * u ** 3)
        b2 = 1.0 / 6.0 * (u ** 3)

        # Calculate interpolated poses
        A0 = b0 * pose1.rminus(pose0)
        A1 = b1 * pose2.rminus(pose1)
        A2 = b2 * pose3.rminus(pose2)

        # Finally get the interpolated pose
        pose_interp = pose0 + A0 + A1 + A2

        return True, pose_interp


    def get_velocity(self, timestamp):
        """
        Get the velocity at a specific timestamp.

        :param timestamp: Desired timestamp for which to get the velocity.
        :return: True if successful, False otherwise.
        """
        # Get the bounding poses for the desired timestamp
        t0, t1, t2, t3 = -1, -1, -1, -1
        pose0, pose1, pose2, pose3 = None, None, None, None

        success, t0, pose0, t1, pose1, t2, pose2, t3, pose3 = self.find_bounding_control_points(timestamp)

        # Return failure if we can't get bounding poses
        if not success:
            return False, None, None

        # Our De Boor-Cox matrix scalars
        DT = (t2 - t1)
        u = (timestamp - t1) / DT
        b0 = 1.0 / 6.0 * (5 + 3 * u - 3 * u ** 2 + u ** 3)
        b1 = 1.0 / 6.0 * (1 + 3 * u + 3 * u ** 2 - 2 * u ** 3)
        b2 = 1.0 / 6.0 * (u ** 3)
        b0dot = 1.0 / (6.0 * DT) * (3 - 6 * u + 3 * u ** 2)
        b1dot = 1.0 / (6.0 * DT) * (3 + 6 * u - 6 * u ** 2)
        b2dot = 1.0 / (6.0 * DT) * (3 * u ** 2)

        # Cache some values we use a lot
        omega_10 = pose1.rminus(pose0)
        omega_21 = pose2.rminus(pose1)
        omega_32 = pose3.rminus(pose2)

        # Calculate interpolated poses
        A0 = (b0 * omega_10).exp()
        A1 = (b1 * omega_21).exp()
        A2 = (b2 * omega_32).exp()

        A0dot = b0dot * omega_10.hat() @ A0.transform()
        A1dot = b1dot * omega_21.hat() @ A1.transform()
        A2dot = b2dot * omega_32.hat() @ A2.transform()

        # Get the interpolated pose
        pose_interp = pose0*A0*A1*A2 # The same as 'compose' operation
        # print(omega_10.hat() @ A0.transform())
        # Keep in mind that Temp value is not in the Lie group maniford.
        # print(A0dot @ A1.transform() @ A2.transform())
        # print(A0.transform() @ A1dot @ A2.transform())
        # print(A0.transform() @ A1.transform() @ A2dot)
        Tangent_vector_on_pose = A0dot @ A1.transform() @ A2.transform() + A0.transform() @ A1dot @ A2.transform() + A0.transform() @ A1.transform() @ A2dot

        # Finally get the interpolated velocities
        vel_interp = pose0.transform() @ Tangent_vector_on_pose
        # vel_interp = pose0.transform() @ lie.SE2Tangent(Tangent_vector_on_pose[0, 2], Tangent_vector_on_pose[1, 2], Tangent_vector_on_pose[1, 0]).hat()
        # vel = (pose_interp.inverse().transform() @ vel_interp)
        # x = vel[0, 2]
        # y = vel[1, 2]
        # yaw = vel[1, 0]
        # print(vel)

        # print(inv(pose_interp.transform()) @ vel_interp)
        # print((lie.SE2(Temp[0, 2], Te, rotation_angle_2d(Temp[:2, :2]))).transform())
        return True, pose_interp, lie.SE2Tangent(vel_interp[0, 2], vel_interp[1, 2], vel_interp[1, 0])


    # def get_acceleration(self, timestamp, R_GtoI, p_IinG, w_IinI, v_IinG, alpha_IinI, a_IinG):
    #     # Get the bounding poses for the desired timestamp
    #     t0, t1, t2, t3 = 0, 0, 0, 0
    #     pose0, pose1, pose2, pose3 = np.eye(4), np.eye(4), np.eye(4), np.eye(4)
    #     success = self.find_bounding_control_points(timestamp, self.control_points, t0, pose0, t1, pose1, t2, pose2, t3, pose3)

    #     # Return failure if we can't get bounding poses
    #     if not success:
    #         alpha_IinI.fill(0)
    #         a_IinG.fill(0)
    #         return False

    #     # Our De Boor-Cox matrix scalars
    #     DT = t2 - t1
    #     u = (timestamp - t1) / DT
    #     b0 = (1.0 / 6.0) * (5 + 3 * u - 3 * u * u + u * u * u)
    #     b1 = (1.0 / 6.0) * (1 + 3 * u + 3 * u * u - 2 * u * u * u)
    #     b2 = (1.0 / 6.0) * (u * u * u)
    #     b0dot = (1.0 / (6.0 * DT)) * (3 - 6 * u + 3 * u * u)
    #     b1dot = (1.0 / (6.0 * DT)) * (3 + 6 * u - 6 * u * u)
    #     b2dot = (1.0 / (6.0 * DT)) * (3 * u * u)
    #     b0dotdot = (1.0 / (6.0 * DT * DT)) * (-6 + 6 * u)
    #     b1dotdot = (1.0 / (6.0 * DT * DT)) * (6 - 12 * u)
    #     b2dotdot = (1.0 / (6.0 * DT * DT)) * (6 * u)

    #     # Cache some values we use a lot
    #     omega_10 = self.log_se3(self.inv_se3(pose0) @ pose1)
    #     omega_21 = self.log_se3(self.inv_se3(pose1) @ pose2)
    #     omega_32 = self.log_se3(self.inv_se3(pose2) @ pose3)
    #     omega_10_hat = self.hat_se3(omega_10)
    #     omega_21_hat = self.hat_se3(omega_21)
    #     omega_32_hat = self.hat_se3(omega_32)

    #     # Calculate interpolated poses
    #     A0 = self.exp_se3(b0 * omega_10)
    #     A1 = self.exp_se3(b1 * omega_21)
    #     A2 = self.exp_se3(b2 * omega_32)
    #     A0dot = b0dot * omega_10_hat @ A0
    #     A1dot = b1dot * omega_21_hat @ A1
    #     A2dot = b2dot * omega_32_hat @ A2
    #     A0dotdot = b0dot * omega_10_hat @ A0dot + b0dotdot * omega_10_hat @ A0
    #     A1dotdot = b1dot * omega_21_hat @ A1dot + b1dotdot * omega_21_hat @ A1
    #     A2dotdot = b2dot * omega_32_hat @ A2dot + b2dotdot * omega_32_hat @ A2

    #     # Get the interpolated pose
    #     pose_interp = pose0 @ A0 @ A1 @ A2
    #     R_GtoI[:] = pose_interp[:3, :3].T
    #     p_IinG[:] = pose_interp[:3, 3]

    #     # Get the interpolated velocities
    #     # NOTE: Rdot = R*skew(omega) => R^T*Rdot = skew(omega)
    #     vel_interp = pose0 @ (A0dot @ A1 @ A2 + A0 @ A1dot @ A2 + A0 @ A1 @ A2dot)
    #     w_IinI[:] = self.vee(pose_interp[:3, :3].T @ vel_interp[:3, :3])
    #     v_IinG[:] = vel_interp[:3, 3]

    #     # Finally get the interpolated accelerations
    #     acc_interp = pose0 @ (A0dotdot @ A1 @ A2 + A0 @ A1dotdot @ A2 + A0 @ A1 @ A2dotdot +
    #                           2 * A0dot @ A1dot @ A2 + 2 * A0 @ A1dot @ A2dot + 2 * A0dot @ A1 @ A2dot)
    #     omegaskew = pose_interp[:3, :3].T @ vel_interp[:3, :3]
    #     alpha_IinI[:] = self.vee(pose_interp[:3, :3].T @ (acc_interp[:3, :3] - vel_interp[:3, :3] @ omegaskew))
    #     a_IinG[:] = acc_interp[:3, 3]
    #     return True

    def get_start_time(self):
        """Returns the simulation start time that we should start simulating from."""
        return self.timestamp_start


