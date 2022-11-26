#!/usr/bin/env python3

#==============================================================================
# File name          : gem_gnss_tracker_stanley_rtk.py                                                                  
# Description        : gnss waypoints tracker using pid and Stanley controller                                                              
# Author             : Hang Cui (hangcui3@illinois.edu)                                       
# Date created       : 08/08/2022                                                                 
# Date last modified : 08/18/2022                                                          
# Version            : 1.0                                                                    
# Usage              : rosrun gem_gnss_control gem_gnss_tracker_stanley_rtk.py                                                                      
# Python version     : 3.8   
# Longitudinal ctrl  : Ji'an Pan (pja96@illinois.edu), Peng Hang (penghan2@illinois.edu)                                                            
#==============================================================================

from __future__ import print_function

# Python Headers
import os 
import csv
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal

# ROS Headers
import rospy
import alvinxy.alvinxy as axy 
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import String, Bool, Float32, Float64
from novatel_gps_msgs.msg import NovatelPosition, NovatelXYZ, Inspva
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from gem_vision.msg import waypoint


# GEM PACMod Headers
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt


class Onlinct_errorilter(object):

    def __init__(self, cutoff, fs, order):
        
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq

        # Get the filter coct_errorficients 
        self.b, self.a = signal.butter(order, normal_cutoff, btype='low', analog=False)

        # Initialize
        self.z = signal.lfilter_zi(self.b, self.a)
    
    def get_data(self, data):

        filted, self.z = signal.lfilter(self.b, self.a, [data], zi=self.z)
        return filted


class PID(object):

    def __init__(self, kp, ki, kd, wg=None):

        self.iterm  = 0
        self.last_t = None
        self.last_e = 0
        self.kp     = kp
        self.ki     = ki
        self.kd     = kd
        self.wg     = wg
        self.derror = 0

    def reset(self):
        self.iterm  = 0
        self.last_e = 0
        self.last_t = None

    def get_control(self, t, e, fwd=0):

        if self.last_t is None:
            self.last_t = t
            de = 0
        else:
            de = (e - self.last_e) / (t - self.last_t)

        if abs(e - self.last_e) > 0.5:
            de = 0

        self.iterm += e * (t - self.last_t)

        # take care of integral winding-up
        if self.wg is not None:
            if self.iterm > self.wg:
                self.iterm = self.wg
            elif self.iterm < -self.wg:
                self.iterm = -self.wg

        self.last_e = e
        self.last_t = t
        self.derror = de

        return fwd + self.kp * e + self.ki * self.iterm + self.kd * de


class Stanley(object):
    
    def __init__(self):

        self.rate   = rospy.Rate(30)

        self.olat   = 40.0928232 
        self.olon   = -88.2355788

        self.offset = 1.1 # meters

        # PID for longitudinal control
        self.desired_speed = 0.6  # m/s
        self.max_accel     = 0.48 # % of acceleration
        self.pid_speed     = PID(0.5, 0.0, 0.1, wg=20)
        self.speed_filter  = Onlinct_errorilter(1.2, 30, 4)

        self.gnss_sub   = rospy.Subscriber("/novatel/inspva", Inspva, self.inspva_callback)
        self.lat        = 0.0
        self.lon        = 0.0
        self.heading    = 0.0

        self.speed_sub  = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)
        self.speed      = 0.0

        self.stanley_pub = rospy.Publisher('/gem/stanley_gnss_cmd', AckermannDrive, queue_size=1)

        self.waypoint = rospy.Subscriber('waypoint', waypoint, self.waypoint_callback)


        # -------------------- PACMod setup --------------------

        self.enable_sub = rospy.Subscriber("/pacmod/as_tx/enable", Bool, self.enable_callback)

        self.gem_enable    = False
        self.pacmod_enable = False

        # GEM vehicle enable, publish once
        self.enable_pub = rospy.Publisher('/pacmod/as_rx/enable', Bool, queue_size=1)
        self.enable_cmd = Bool()
        self.enable_cmd.data = False

        # GEM vehicle gear control, neutral, forward and reverse, publish once
        self.gear_pub = rospy.Publisher('/pacmod/as_rx/shift_cmd', PacmodCmd, queue_size=1)
        self.gear_cmd = PacmodCmd()
        self.gear_cmd.ui16_cmd = 2 # SHIFT_NEUTRAL

        # GEM vehilce brake control
        self.brake_pub = rospy.Publisher('/pacmod/as_rx/brake_cmd', PacmodCmd, queue_size=1)
        self.brake_cmd = PacmodCmd()
        self.brake_cmd.enable = False
        self.brake_cmd.clear  = True
        self.brake_cmd.ignore = True

        # GEM vechile forward motion control
        self.accel_pub = rospy.Publisher('/pacmod/as_rx/accel_cmd', PacmodCmd, queue_size=1)
        self.accel_cmd = PacmodCmd()
        self.accel_cmd.enable = False
        self.accel_cmd.clear  = True
        self.accel_cmd.ignore = True

        # GEM vechile turn signal control
        self.turn_pub = rospy.Publisher('/pacmod/as_rx/turn_cmd', PacmodCmd, queue_size=1)
        self.turn_cmd = PacmodCmd()
        self.turn_cmd.ui16_cmd = 1 # None

        # GEM vechile steering wheel control
        self.steer_pub = rospy.Publisher('/pacmod/as_rx/steer_cmd', PositionWithSpeed, queue_size=1)
        self.steer_cmd = PositionWithSpeed()
        self.steer_cmd.angular_position = 0.0 # radians, -: clockwise, +: counter-clockwise
        self.steer_cmd.angular_velocity_limit = 2.0 # radians/second

        self.ackermann_msg                         = AckermannDrive()
        self.ackermann_msg.steering_angle_velocity = 0.0
        self.ackermann_msg.acceleration            = 0.0
        self.ackermann_msg.jerk                    = 0.0
        self.ackermann_msg.speed                   = 0.0 
        self.ackermann_msg.steering_angle          = 0.0

        self.waypoint_x_1 = None
        self.waypoint_y_1 = None
        self.waypoint_x_2 = None
        self.waypoint_y_2 = None

        # read waypoints into the system           
        self.read_waypoints() 

        # Hang 
        self.steer = 0.0 # degrees
        self.steer_sub = rospy.Subscriber("/pacmod/parsed_tx/steer_rpt", SystemRptFloat, self.steer_callback)


    # Get GNSS information
    def inspva_callback(self, inspva_msg):
        self.lat     = inspva_msg.latitude  # latitude
        self.lon     = inspva_msg.longitude # longitude
        self.heading = inspva_msg.azimuth   # heading in degrees

    def waypoint_callback(self, msg):
        self.waypoint_x_1 = msg.x_1
        self.waypoint_y_1 = msg.y_1
        self.waypoint_x_2 = msg.x_2
        self.waypoint_y_2 = msg.y_2

    def enable_callback(self, msg):
        self.pacmod_enable = msg.data

    # Get vehicle speed
    def speed_callback(self, msg):
        self.speed = round(msg.vehicle_speed, 3) # forward velocity in m/s


    # Get value of steering wheel
    def steer_callback(self, msg):
        self.steer = round(np.degrees(msg.output),1)


    # Get predefined waypoints based on GNSS
    def read_waypoints(self):

        # read recorded GPS lat, lon, heading
        dirname  = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../waypoints/xyhead_demo_stanley.csv')

        with open(filename) as f:
            path_points = [tuple(line) for line in csv.reader(f)]

        # x towards East and y towards North
        self.path_points_lon_x   = [float(point[0]) for point in path_points] # longitude
        self.path_points_lat_y   = [float(point[1]) for point in path_points] # latitude
        self.path_points_heading = [float(point[2]) for point in path_points] # heading

    # Conversion of front wheel to steering wheel
    def front2steer(self, f_angle):
        if(f_angle > 35):
            f_angle = 35
        if (f_angle < -35):
            f_angle = -35
        if (f_angle > 0):
            steer_angle = round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        elif (f_angle < 0):
            f_angle = -f_angle
            steer_angle = -round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        else:
            steer_angle = 0.0
        return steer_angle


    # Conversion of Lon & Lat to X & Y
    def wps_to_local_xy_stanley(self, lon_wp, lat_wp):
        # convert GNSS waypoints into local fixed frame reprented in x and y
        lon_wp_x, lat_wp_y = axy.ll2xy(lat_wp, lon_wp, self.olat, self.olon)
        return -lon_wp_x, -lat_wp_y   


    # Conversion of GNSS heading to vehicle heading
    def heading_to_yaw_stanley(self, heading_curr):
        if (heading_curr >= 0 and heading_curr < 90):
            yaw_curr = np.radians(-heading_curr-90)
        else:
            yaw_curr = np.radians(-heading_curr+270)
        return yaw_curr


    # Get vehicle states: x, y, yaw
    def get_gem_state(self):

        # vehicle gnss heading (yaw) in degrees
        # vehicle x, y position in fixed local frame, in meters
        # rct_errorerence point is located at the center of GNSS antennas
        local_x_curr, local_y_curr = self.wps_to_local_xy_stanley(self.lon, self.lat)

        # heading to yaw (degrees to radians)
        # heading is calculated from two GNSS antennas
        curr_yaw = self.heading_to_yaw_stanley(self.heading) 

        # rct_errorerence point is located at the center of front axle
        curr_x = local_x_curr + self.offset * np.cos(curr_yaw)
        curr_y = local_y_curr + self.offset * np.sin(curr_yaw)

        return round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)


    # Find close yaw in predefined GNSS waypoint list
    def find_close_yaw(self, arr, val):
        diff_arr = np.array( np.abs( np.abs(arr) - np.abs(val) ) )
        idx = np.where(diff_arr < 0.5)
        return idx


    # Conversion to -pi to pi
    def pi_2_pi(self, angle):

        if angle > np.pi:
            return angle - 2.0 * np.pi

        if angle < -np.pi:
            return angle + 2.0 * np.pi

        return angle

    # Computes the Euclidean distance between two 2D points
    def dist(self, p1, p2):
        return round(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 3)

    # Start Stanley controller
    def start_stanley(self):
        
        while not rospy.is_shutdown():

            if (self.gem_enable == False):

                if(self.pacmod_enable == True):

                    # ---------- enable PACMod ----------

                    # enable forward gear
                    self.gear_cmd.ui16_cmd = 3

                    # enable brake
                    self.brake_cmd.enable  = True
                    self.brake_cmd.clear   = False
                    self.brake_cmd.ignore  = False
                    self.brake_cmd.f64_cmd = 0.0

                    # enable gas 
                    self.accel_cmd.enable  = True
                    self.accel_cmd.clear   = False
                    self.accel_cmd.ignore  = False
                    # self.accel_cmd.f64_cmd = 0.0

                    self.gear_pub.publish(self.gear_cmd)
                    print("Foward Engaged!")

                    self.turn_pub.publish(self.turn_cmd)
                    print("Turn Signal Ready!")
                    
                    self.brake_pub.publish(self.brake_cmd)
                    print("Brake Engaged!")

                    self.accel_pub.publish(self.accel_cmd)
                    print("Gas Engaged!")

                    self.gem_enable = True
                    # self.enable_pub.publish(self.enable_cmd)



            k = 0.4
            # coordinates of rct_errorerence point (center of frontal axle) in global frame
            curr_x, curr_y, curr_yaw = self.get_gem_state()

            if self.waypoint_x_1 is None:
                continue


            # print(self.waypoint_x_1)
            # print("curr_x", curr_x)
            # print("curr_y", curr_y)

            # front_x = self.wheelbase*np.cos(curr_yaw) + curr_x
            # front_y = self.wheelbase*np.sin(curr_yaw) + curr_y

            # --------------------------- Longitudinal control using PD controller ---------------------------

            filt_vel = np.squeeze(self.speed_filter.get_data(self.speed))
            a_expected = self.pid_speed.get_control(rospy.get_time(), self.desired_speed - filt_vel)

            if a_expected > 0.64 :
                throttle_percent = 0.5

            if a_expected < 0.0 :
                throttle_percent = 0.0

            throttle_percent = (a_expected+2.3501) / 7.3454

            if throttle_percent > self.max_accel:
                throttle_percent = self.max_accel

            if throttle_percent < 0.3:
                throttle_percent = 0.37

            # -------------------------------------- Stanley controller --------------------------------------

            error_num = ((self.waypoint_x_2 - self.waypoint_x_1) * (self.waypoint_y_1 - curr_y)) - ((self.waypoint_x_1 - curr_x) * (self.waypoint_y_2 - self.waypoint_y_1))
            error_denom = np.sqrt((self.waypoint_x_2 - self.waypoint_x_1)**2 + (self.waypoint_y_2 - self.waypoint_y_1)**2)
            error = error_num/error_denom

            theta = np.arctan2(self.waypoint_y_2 - self.waypoint_y_1, self.waypoint_x_2 - self.waypoint_x_1)
            heading_error = theta - curr_yaw
            steering_correction = np.arctan2(filt_vel, k*error)
            steering = np.round(np.clip(self.pi_2_pi(heading_error + steering_correction), -0.61, 0.61), 3)
            steering_degrees = np.degrees(steering)
            steering_angle = self.front2steer(steering_degrees)

            print("steering", steering)
            print("steering angle", steering_angle)
            print("throttle_percenT ", throttle_percent)

            # self.accel_cmd.f64_cmd = throttle_percent
            # self.steer_cmd.angular_position = np.radians(steering_angle)
            # self.accel_pub.publish(self.accel_cmd)
            # self.steer_pub.publish(self.steer_cmd)
            # self.turn_pub.publish(self.turn_cmd)

            self.turn_cmd.ui16_cmd = 2 # turn left

            self.enable_pub.publish(self.enable_cmd)
            self.gear_pub.publish(self.gear_cmd)
            self.brake_pub.publish(self.brake_cmd)
            self.turn_pub.publish(self.turn_cmd)

            # self.brake_pub.publish(self.brake_cmd)

            if (filt_vel < 0.2):
                # self.accel_cmd.f64_cmd = throttle_percent
                self.steer_cmd.angular_position  = 0
                # print(self.ackermann_msg.steering_angle)
            else:
                # self.accel_cmd.f64_cmd    = throttle_percent
                self.steer_cmd.angular_position = round(steering_angle,1)
                # print(self.ackermann_msg.steering_angle)

            
            self.accel_cmd.f64_cmd = 0.36

            self.accel_pub.publish(self.accel_cmd)
            self.steer_pub.publish(self.steer_cmd)

            # ------------------------------------------------------------------------------------------------ 

            # print(self.ackermann_msg.acceleration, self.ackermann_msg.steering_angle)
            # self.stanley_pub.publish(self.ackermann_msg)

            self.rate.sleep()


def stanley_run():

    rospy.init_node('gnss_stanley_node', anonymous=True)
    stanley = Stanley()

    try:
        stanley.start_stanley()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    stanley_run()
