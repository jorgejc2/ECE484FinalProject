from gem_gnss_tracker_stanley_rtk import *

class Stanley_Camera(Stanley):

    def __init__(self):
        super().__init__()

    def start_stanley(self):
        while not rospy.is_shutdown():

            self.path_points_x   = np.array(self.path_points_lon_x)
            self.path_points_y   = np.array(self.path_points_lat_y)
            self.path_points_yaw = np.array(self.path_points_heading)

            # coordinates of rct_errorerence point (center of frontal axle) in global frame
            curr_x, curr_y, curr_yaw = self.get_gem_state()

            # print("X,Y,Yaw: ", curr_x, curr_y, curr_yaw)

            target_idx = self.find_close_yaw(self.path_points_yaw, curr_yaw)

            # print("Target list", target_idx)

            self.target_path_points_x   = self.path_points_x[target_idx]
            self.target_path_points_y   = self.path_points_y[target_idx]
            self.target_path_points_yaw = self.path_points_yaw[target_idx]

            # find the closest point
            dx = [curr_x - x for x in self.target_path_points_x]
            dy = [curr_y - y for y in self.target_path_points_y]

            # find the index of closest point
            target_point_idx = int(np.argmin(np.hypot(dx, dy)))


            if (target_point_idx != len(self.target_path_points_x) -1):
                target_point_idx = target_point_idx + 1


            vec_target_2_front    = np.array([[dx[target_point_idx]], [dy[target_point_idx]]])
            front_axle_vec_rot_90 = np.array([[np.cos(curr_yaw - np.pi / 2.0)], [np.sin(curr_yaw - np.pi / 2.0)]])

            # print("T_X,T_Y,T_Yaw: ", self.target_path_points_x[target_point_idx], \
            #                          self.target_path_points_y[target_point_idx], \
            #                          self.target_path_points_yaw[target_point_idx])

            # crosstrack error
            ct_error = np.dot(vec_target_2_front.T, front_axle_vec_rot_90)
            ct_error = float(np.squeeze(ct_error))

            # heading error
            theta_e = self.pi_2_pi(self.target_path_points_yaw[target_point_idx]-curr_yaw) 

            # theta_e = self.target_path_points_yaw[target_point_idx]-curr_yaw 
            theta_e_deg = round(np.degrees(theta_e), 1)
            print("Crosstrack Error: " + str(round(ct_error,3)) + ", Heading Error: " + str(theta_e_deg))

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

            f_delta        = round(theta_e + np.arctan2(ct_error*0.4, filt_vel), 3)
            f_delta        = round(np.clip(f_delta, -0.61, 0.61), 3)
            f_delta_deg    = np.degrees(f_delta)
            steering_angle = self.front2steer(f_delta_deg)

            if (filt_vel < 0.2):
                self.ackermann_msg.acceleration   = throttle_percent
                self.ackermann_msg.steering_angle = 0
                print(self.ackermann_msg.steering_angle)
            else:
                self.ackermann_msg.acceleration   = throttle_percent
                self.ackermann_msg.steering_angle = round(steering_angle,1)
                print(self.ackermann_msg.steering_angle)

            # ------------------------------------------------------------------------------------------------ 

            self.stanley_pub.publish(self.ackermann_msg)

            self.rate.sleep()