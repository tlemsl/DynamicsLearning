#!/usr/bin/env python

import rospy
import csv
import time
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from message_filters import Subscriber, ApproximateTimeSynchronizer

class DataSaver:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('data_saver')

        # Create subscribers
        self.odom_sub = Subscriber('/synced_odom', Odometry)
        self.cmd_sub = Subscriber('/base_board/controller_cmd', TwistStamped)

        # Synchronize the messages with an approximate time policy
        self.ats = ApproximateTimeSynchronizer(
            [self.odom_sub, self.cmd_sub],
            queue_size=100,
            slop=0.1
        )
        self.ats.registerCallback(self.callback)

        # Open CSV file for writing
        self.csv_file = open('free_drive.csv', 'w')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'x', 'y', 'z', 'yaw','v_horizontal','v_vertical','accel', 'steer'])
        self.prev_time = time.time()
        rospy.loginfo("Data saver node started.")
        rospy.spin()

    def callback(self, odom_msg, cmd_msg):
        # Extract the timestamp (using odom_msg's timestamp as the reference)
        timestamp = odom_msg.header.stamp.to_sec()

        # Extract data from Odometry message
        x = odom_msg.pose.pose.position.x
        y = odom_msg.pose.pose.position.y
        z = odom_msg.pose.pose.position.z
        v_hor = odom_msg.twist.twist.linear.x
        v_vert = odom_msg.twist.twist.linear.y
        # Assuming the orientation is given in quaternion and converting to yaw
        _, _, yaw = self.quaternion_to_euler(odom_msg.pose.pose.orientation)

        # Extract data from TwistStamped message
        accel = cmd_msg.twist.linear.x
        steer = cmd_msg.twist.angular.z

        # Write data to CSV
        now = time.time()
        self.csv_writer.writerow([timestamp, x, y, z, yaw,v_hor, v_vert, accel, steer])
        print(f"Curr HZ {1/(now - self.prev_time)}")
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

if __name__ == '__main__':
    DataSaver()

