from BsplineLie import BsplineLie
import manifpy as lie
import numpy as np
from matplotlib import pyplot as plt

spline = BsplineLie()
poses = []
HZ = 10
dt = 1 / HZ
endtime = 10
t = 0

# Parameters for the circular trajectory
radius = 5       # Radius of the circle
center = np.array([0, 0])  # Center of the circle (optional)
angular_speed = 2 * np.pi / endtime  # One full circle over the total time
X_ref = []
# Generate poses along the circular path
while t < endtime:
    x = center[0] + radius * np.cos(angular_speed * t)
    y = center[1] + radius * np.sin(angular_speed * t)
    theta = angular_speed * t  # Orientation following the circular path
    X_ref.append([x,y,theta])
    pose = lie.SE2(x, y, theta)  # Create SE(2) pose with translation and rotation
    poses.append([t, pose])  # Store the pose with timestamp
    t += dt

spline.feed_trajectory(poses)

# Collect interpolated poses and their yaw angles
dense_timestamps = np.arange(0, endtime, dt)
interpolated_poses = []
interpolated_vels = []

for t in dense_timestamps:
    success, pose, vel = spline.get_velocity(t)
    if success:
        interpolated_poses.append([pose.x(), pose.y(), pose.angle()])
        interpolated_vels.append([vel.x(), vel.y(), vel.angle()])
        vel_ref = np.array([-radius * angular_speed * np.sin(pose.angle()), radius * angular_speed * np.cos(pose.angle()), angular_speed])
        
        print(f"\n\nref: {vel_ref}")
        print(f"dot: {interpolated_vels[-1]}")
        print(f"dif: {interpolated_vels[-1] - vel_ref}")
        
interpolated_poses = np.array(interpolated_poses)
interpolated_vels = np.array(interpolated_vels)


X_ref = np.array(X_ref)
# Plot the trajectory and yaw angle
fig, (ax1) = plt.subplots(1, 1, figsize=(10, 10))

# Extract X, Y, and yaw values from interpolated data
x_vals = interpolated_poses[:, 0]
y_vals = interpolated_poses[:, 1]


# 2D Trajectory plot with arrows for both interpolated and reference yaw
ax1.scatter(X_ref[:, 0], X_ref[:, 1], color='red', marker='o', label='Original Poses')
# ax1.plot(x_vals, y_vals, linestyle='--', color='blue', marker = 'o', label='Interpolated Path')

# Arrow scale for visual clarity
arrow_scale = 0.5
# Plot arrows for interpolated velocities
#  -radius * angular_speed * np.sin(angular_speed * t),
#  radius * angular_speed * np.cos(angular_speed * t),
for i in range(len(interpolated_poses)):
    x, y, yaw = interpolated_poses[i]
    vx, vy, _ = interpolated_vels[i]
    theta = np.arctan2(vx, vy)
    ax1.arrow(
        x, y,vx, vy,
        head_width=0.2, head_length=0.3, fc='orange', ec='orange'
    )

# # Plot interpolated yaw arrows (in green)
# for x, y, theta in zip(x_vals, y_vals, yaw_vals):
#     ax1.arrow(
#         x, y,
#         arrow_scale * np.cos(theta), arrow_scale * np.sin(theta),
#         head_width=0.2, head_length=0.3, fc='green', ec='green'
#     )

# # Plot reference yaw arrows (in orange)
# for x, y, theta_ref in X_ref:
#     ax1.arrow(
#         x, y,
#         arrow_scale * np.cos(theta_ref), arrow_scale * np.sin(theta_ref),
#         head_width=0.2, head_length=0.3, fc='orange', ec='orange'
#     )
ax1.set_title("Circular SE(2) Pose Interpolation with Interpolated and Ref Yaw")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.legend(['Interpolated Path'])
ax1.grid(True)
# ax1.axis('equal')

# # Yaw Angle vs Time plot
# ax2.plot(interpolated_yaws[:, 0], ref_yaw_vals, color='orange', label='Ref Yaw Angle (Theta)')
# ax2.plot(interpolated_yaws[:, 0], yaw_vals, color='green', label='Interpolated Yaw Angle (Theta)')
# ax2.set_title("Yaw Angle (Theta) over Time")
# ax2.set_xlabel("Time (s)")
# ax2.set_ylabel("Yaw Angle (radians)")
# ax2.legend()
# ax2.grid(True)

# plt.tight_layout()
plt.show()
