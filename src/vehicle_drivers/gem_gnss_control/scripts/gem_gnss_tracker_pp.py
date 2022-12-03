#!/usr/bin/env python3

#================================================================
# File name: gem_gnss_pp_tracker_pid.py                                                                  
# Description: gnss waypoints tracker using pid and pure pursuit                                                                
# Author: Hang Cui
# Email: hangcui3@illinois.edu                                                                     
# Date created: 08/02/2021                                                                
# Date last modified: 08/15/2022                                                          
# Version: 1.0                                                                   
# Usage: rosrun gem_gnss gem_gnss_pp_tracker.py                                                                      
# Python version: 3.8                                                             
#================================================================

from __future__ import print_function

# Python Headers
import os 
import csv
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal
import json

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
from lidarProcessing import LidarProcessing


# GEM Sensor Headers
from std_msgs.msg import String, Bool, Float32, Float64
from novatel_gps_msgs.msg import NovatelPosition, NovatelXYZ, Inspva

# GEM PACMod Headers
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt


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

        # data for post processing
        self.post_gnss_lat = []
        self.post_gnss_long = []
        self.post_gnss_heading = []
        self.post_published_acceleration = []
        self.post_published_heading = []
        self.post_pacmod_velocity = []
        self.post_velodyne_lidar = []
        self.post_waypoint_x_1 = []
        self.post_waypoint_y_1 = []
        self.post_waypoint_x_2 = []
        self.post_waypoint_y_2 = []

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


class OnlineFilter(object):

    def __init__(self, cutoff, fs, order):
        
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq

        # Get the filter coefficients 
        self.b, self.a = signal.butter(order, normal_cutoff, btype='low', analog=False)

        # Initialize
        self.z = signal.lfilter_zi(self.b, self.a)
    
    def get_data(self, data):
        filted, self.z = signal.lfilter(self.b, self.a, [data], zi=self.z)
        return filted


class PurePursuit(object):
    
    def __init__(self):

        self.rate       = rospy.Rate(10)

        self.look_ahead = 4
        self.wheelbase  = 1.75 # meters
        self.offset     = 0.50 # meters originally 0.46 from GNSS

        self.gnss_sub   = rospy.Subscriber("/novatel/inspva", Inspva, self.inspva_callback)
        self.lat        = 0.0
        self.lon        = 0.0
        self.heading    = 0.0

        self.enable_sub = rospy.Subscriber("/pacmod/as_tx/enable", Bool, self.enable_callback)

        self.waypoint = rospy.Subscriber('waypoint', waypoint, self.waypoint_callback)

        self.speed_sub  = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)
        self.speed      = 0.0

        self.olat       = 40.0928563
        self.olon       = -88.2359994

        # read waypoints into the system 
        self.goal       = 0            
        self.read_waypoints() 

        self.desired_speed = 1.25  # m/s, reference speed
        self.max_accel     = 0.48 # % of acceleration
        self.pid_speed     = PID(0.5, 0.0, 0.1, wg=20)
        self.speed_filter  = OnlineFilter(1.2, 30, 4)

        # -------------------- PACMod setup --------------------

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


        self.waypoint_x_1 = None
        self.waypoint_y_1 = None
        self.waypoint_x_2 = None
        self.waypoint_y_2 = None

        # lidar
        self.lidar_reading = LidarProcessing()


    def inspva_callback(self, inspva_msg):
        self.lat     = inspva_msg.latitude  # latitude
        self.lon     = inspva_msg.longitude # longitude
        self.heading = inspva_msg.azimuth   # heading in degrees
        # print('yaw test')

    def speed_callback(self, msg):
        self.speed = round(msg.vehicle_speed, 3) # forward velocity in m/s

    def waypoint_callback(self, msg):
        self.waypoint_x_1 = msg.x_1
        self.waypoint_y_1 = msg.y_1
        self.waypoint_x_2 = msg.x_2
        self.waypoint_y_2 = msg.y_2

    def enable_callback(self, msg):
        self.pacmod_enable = msg.data

    def heading_to_yaw(self, heading_curr):
        if (heading_curr >= 270 and heading_curr < 360):
            yaw_curr = np.radians(450 - heading_curr)
        else:
            yaw_curr = np.radians(90 - heading_curr)
        return yaw_curr

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

    def read_waypoints(self):
        # read recorded GPS lat, lon, heading
        dirname  = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../waypoints/xyhead_demo_pp.csv')
        with open(filename) as f:
            path_points = [tuple(line) for line in csv.reader(f)]
        # x towards East and y towards North
        self.path_points_lon_x   = [float(point[0]) for point in path_points] # longitude
        self.path_points_lat_y   = [float(point[1]) for point in path_points] # latitude
        self.path_points_heading = [float(point[2]) for point in path_points] # heading
        self.wp_size             = len(self.path_points_lon_x)
        self.dist_arr            = np.zeros(self.wp_size)

    def wps_to_local_xy(self, lon_wp, lat_wp):
        # convert GNSS waypoints into local fixed frame reprented in x and y
        lon_wp_x, lat_wp_y = axy.ll2xy(lat_wp, lon_wp, self.olat, self.olon)
        return lon_wp_x, lat_wp_y   

    def get_gem_state(self):

        # vehicle gnss heading (yaw) in degrees
        # vehicle x, y position in fixed local frame, in meters
        # reference point is located at the center of GNSS antennas
        local_x_curr, local_y_curr = self.wps_to_local_xy(self.lon, self.lat)

        # heading to yaw (degrees to radians)
        # heading is calculated from two GNSS antennas
        curr_yaw = self.heading_to_yaw(self.heading) 

        # reference point is located at the center of rear axle
        curr_x = local_x_curr - self.offset * np.cos(curr_yaw)
        curr_y = local_y_curr - self.offset * np.sin(curr_yaw)

        return round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)

    # find the angle bewtween two vectors    
    def find_angle(self, v1, v2):
        cosang = np.dot(v1, v2)
        sinang = la.norm(np.cross(v1, v2))
        # [-pi, pi]
        return np.arctan2(sinang, cosang)

    # computes the Euclidean distance between two 2D points
    def dist(self, p1, p2):
        return round(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 3)

    def start_pp(self):
        
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
                    self.accel_cmd.f64_cmd = 0.0

                    self.gear_pub.publish(self.gear_cmd)
                    print("Foward Engaged!")

                    self.turn_pub.publish(self.turn_cmd)
                    print("Turn Signal Ready!")
                    
                    self.brake_pub.publish(self.brake_cmd)
                    print("Brake Engaged!")

                    self.accel_pub.publish(self.accel_cmd)
                    print("Gas Engaged!")

                    self.gem_enable = True

            curr_x, curr_y, curr_yaw = self.get_gem_state()

            if self.waypoint_x_2 is None:
                continue

            # lidar readings
            
            lidar_reading_value = self.lidar_reading.processLidar()
            safe_breaking_distance = 12  # distance in metres
            print("lidar_reading: ", lidar_reading_value)
            curr_yaw += 90
            print("Curr_yaw: ", curr_yaw)


            # calculate x and y distance from waypoint to center of rear axel
            rear_axel_offsetted_x = self.waypoint_x_2 + self.offset * np.abs(np.cos(curr_yaw * (np.pi/180)))
            rear_axel_offsetted_y = self.waypoint_y_2 + self.offset * np.abs(np.sin(curr_yaw * (np.pi/180)))

            # print("rear_axel_offsetted_x : ", rear_axel_offsetted_x)
            # print("rear_axel_offsetted_y : ", rear_axel_offsetted_y)

            # calculate euclidian distance of vehicle to waypoint
            L = np.sqrt((rear_axel_offsetted_y ** 2) + (rear_axel_offsetted_x ** 2))

            # find angle from waypoint to rear axel
            alpha = np.arctan2(rear_axel_offsetted_x, rear_axel_offsetted_y)

            # print("alpha: ", alpha)

            # goal speed is 1.5
            # look ahead distance = K * goal speed

            # ----------------- tuning this part as needed -----------------
            k       = 0.8 # k value might need to be closer to 1 since goal speed is 1.5 m/s
            angle_i = math.atan((k * 2 * self.wheelbase * math.sin(alpha)) / L) # maybe change this to not have k or k close to 1
            angle   = angle_i*1.2
            # ----------------- tuning this part as needed -----------------

            f_delta = round(np.clip(angle, -0.61, 0.61), 3)

            f_delta_deg = np.degrees(f_delta)
            # print("tire angle: ", f_delta_deg)

            # steering_angle in degrees
            steering_angle = self.front2steer(f_delta_deg)
            print("steering angle: ", steering_angle)

            # if(self.gem_enable == True):
            #     print("Current index: " + str(self.goal))
            #     print("Forward velocity: " + str(self.speed))
            #     ct_error = round(np.sin(alpha) * L, 3)
            #     print("Crosstrack Error: " + str(ct_error))
            #     print("Front steering angle: " + str(np.degrees(f_delta)) + " degrees")
            #     print("Steering wheel angle: " + str(steering_angle) + " degrees" )
            #     print("\n")

            current_time = rospy.get_time()
            filt_vel     = self.speed_filter.get_data(self.speed)
            output_accel = self.pid_speed.get_control(current_time, self.desired_speed - filt_vel)

            if output_accel > self.max_accel:
                output_accel = self.max_accel

            if output_accel < 0.3:
                output_accel = 0.3

            if (f_delta_deg <= 30 and f_delta_deg >= -30):
                self.turn_cmd.ui16_cmd = 1
            elif(f_delta_deg > 30):
                self.turn_cmd.ui16_cmd = 2 # turn left
            else:
                self.turn_cmd.ui16_cmd = 0 # turn rightput_accel
            # self.brake_cmd.f64_cmd = 0.0

            # lidar pedestrian detection stopping
            if (lidar_reading_value[0] <= safe_breaking_distance):
                self.accel_cmd.f64_cmd = 0.0
                self.brake_cmd.f64_cmd = 0.3
            else:
                self.accel_cmd.f64_cmd = output_accel
                self.brake_cmd.f64_cmd = 0.0

            # self.accel_cmd.f64_cmd = output_accel
            # self.brake_cmd.f64_cmd = 0.0
            self.steer_cmd.angular_position = np.radians(steering_angle)
            self.accel_pub.publish(self.accel_cmd)
            self.steer_pub.publish(self.steer_cmd)
            self.turn_pub.publish(self.turn_cmd)
            self.brake_pub.publish(self.brake_cmd)

            # collecting data points for csv file 
            if self.gem_enable == True:
                # car is actually running
                self.post_gnss_lat.append(self.lat)
                self.post_gnss_long.append(self.lon)
                self.post_gnss_heading.append(self.heading)
                self.post_published_acceleration.append(self.accel_cmd)
                self.post_published_heading.append(self.steer_cmd)
                self.post_pacmod_velocity.append(self.speed)
                self.post_velodyne_lidar.append()
                self.post_waypoint_x_1.append(self.waypoint_x_1)
                self.post_waypoint_y_1.append(self.waypoint_y_1)
                self.post_waypoint_x_2.append(self.waypoint_x_2)
                self.post_waypoint_y_2.append(self.waypoint_y_2)


            self.rate.sleep()

        # take data and create json file
        json_data = {
            "gnss_lat" : self.post_gnss_lat,
            "gnss_long" : self.post_gnss_long,
            "gnss_heading" : self.post_gnss_heading,
            "published_acceleration" : self.post_published_acceleration,
            "published_heading" : self.post_published_heading,
            "pacmod_velocity" : self.post_pacmod_velocity,
            "velodyne_lidar" : self.post_velodyne_lidar,
            "x_1" : self.post_waypoint_x_1,
            "x_2" : self.post_waypoint_x_2,
            "y_1" : self.post_waypoint_y_1,
            "y_2" : self.post_waypoint_y_2
        }
        # get list of directories
        curr_dir = os.getcwd()
        dir_list = os.listdir(curr_dir)
        largest_num = None
        # find the largest number in a file name
        for file in dir_list:
            file_num = [x for x in file if x.isdigit()]
            file_num = int("".join(file_num))
            if largest_num is None or (file_num != "" and file_num > largest_num):
                largest_num = file_num
        # write the results to a new file
        largest_num = 0 if largest_num is None else largest_num + 1
        result_file = "result_" + str(largest_num) + ".json"
        with open(result_file, "w") as outfile:
            json.dump(json_data, outfile, indent=4)


def pure_pursuit():

    rospy.init_node('gnss_pp_node', anonymous=True)
    pp = PurePursuit()

    try:
        pp.start_pp()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    pure_pursuit()

