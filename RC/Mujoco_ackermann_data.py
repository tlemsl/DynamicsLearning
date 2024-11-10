#!/usr/bin/env python

import rospy
import csv
import time
import argparse
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from message_filters import Subscriber, ApproximateTimeSynchronizer

class DataSaver:
    def __init__(self, csv_filename):
        # Initialize ROS node
        rospy.init_node('data_saver')

        # Create subscribers
        self.pose_sub = Subscriber('/mushr_mujoco_ros/buddy/pose', PoseStamped)
        self.cmd_sub = Subscriber('/mushr_mujoco_ros/buddy/control', AckermannDriveStamped)

        # Synchronize the messages with an approximate time policy
        self.ats = ApproximateTimeSynchronizer(
            [self.pose_sub, self.cmd_sub],
            queue_size=100,
            slop=0.1
        )
        self.ats.registerCallback(self.callback)

        # Open CSV file for writing
        self.csv_file = open(csv_filename, 'w')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'x', 'y', 'yaw', 'accel', 'steer'])

        self.prev_time = time.time()
        rospy.loginfo("Data saver node started.")
        rospy.spin()

    def callback(self, pose_msg, cmd_msg):
        # Extract the timestamp (using odom_msg's timestamp as the reference)
        timestamp = pose_msg.header.stamp.to_sec()

        # Extract data from Odometry message
        x = pose_msg.pose.position.x
        y = pose_msg.pose.position.y
        
        
        # Assuming the orientation is given in quaternion and converting to yaw
        _, _, yaw = self.quaternion_to_euler(pose_msg.pose.orientation)

        # Extract data from TwistStamped message
        speed = cmd_msg.drive.speed
        steer = cmd_msg.drive.steering_angle

        # Write data to CSV
        now = time.time()
        self.csv_writer.writerow([timestamp, x, y, yaw, speed, steer])
        print(f"Curr HZ {1 / (now - self.prev_time)}")
        self.prev_time = now

    def quaternion_to_euler(self, quaternion):
        # Convert quaternion to Euler angles (yaw)
        import tf.transformations as tft
        _, _, yaw = tft.euler_from_quaternion([
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w
        ])
        return _, _, yaw

    def __del__(self):
        # Close CSV file on exit
        self.csv_file.close()

def parse_args():
    # Set up argparse to get CSV filename from the command line
    parser = argparse.ArgumentParser(description="Save synchronized ROS messages to a CSV file.")
    parser.add_argument('--csv', type=str, required=True, help="CSV filename to save the data.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    DataSaver(args.csv)
