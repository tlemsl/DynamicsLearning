#!/usr/bin/env python3

import rospy
import argparse
from ackermann_msgs.msg import AckermannDriveStamped

def publish_ackermann_cmd(speed, steering_angle, frequency):
    # Initialize ROS node
    rospy.init_node('ackermann_cmd_mockup', anonymous=True)
    
    # Create publisher
    pub = rospy.Publisher('/base_board/cmd', AckermannDriveStamped, queue_size=10)
    
    # Set publishing rate
    rate = rospy.Rate(frequency)
    
    # Create message
    msg = AckermannDriveStamped()
    msg.header.frame_id = "base_link"
    msg.drive.speed = speed
    msg.drive.steering_angle = steering_angle
    
    while not rospy.is_shutdown():
        # Update timestamp
        msg.header.stamp = rospy.Time.now()
        
        # Publish message
        pub.publish(msg)
        
        # Sleep to maintain publishing rate
        rate.sleep()

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Ackermann command mockup publisher')
    parser.add_argument('--speed', type=float, default=0.0,
                        help='Linear velocity (m/s)')
    parser.add_argument('--steering', type=float, default=0.0,
                        help='Steering angle (rad)')
    parser.add_argument('--freq', type=float, default=20.0,
                        help='Publishing frequency (Hz)')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        publish_ackermann_cmd(args.speed, args.steering, args.freq)
    except rospy.ROSInterruptException:
        pass
