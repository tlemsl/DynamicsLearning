#!/usr/bin/env python

import rospy
import random
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import tf

def generate_random_pose():
    pose = PoseStamped()
    pose.header = Header()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = "map"  # Set this to the appropriate frame
    
    # Random position within a 10m x 10m area
    pose.pose.position.x = random.uniform(-5.0, 5.0)
    pose.pose.position.y = random.uniform(-5.0, 5.0)
    pose.pose.position.z = 0.0

    # Random orientation
    yaw = random.uniform(-3.14159, 3.14159)  # Random yaw in radians
    quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
    pose.pose.orientation.x = quaternion[0]
    pose.pose.orientation.y = quaternion[1]
    pose.pose.orientation.z = quaternion[2]
    pose.pose.orientation.w = quaternion[3]

    return pose

def goal_publisher():
    rospy.init_node('goal_publisher', anonymous=True)
    pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
    rate = rospy.Rate(0.3)  # 0.5 Hz (once every 2 seconds)

    while not rospy.is_shutdown():
        random_pose = generate_random_pose()
        pub.publish(random_pose)
        rospy.loginfo(f"Published new goal: Position=({random_pose.pose.position.x:.2f}, {random_pose.pose.position.y:.2f}), Orientation={random_pose.pose.orientation}")
        rate.sleep()

if __name__ == '__main__':
    try:
        goal_publisher()
    except rospy.ROSInterruptException:
        pass

