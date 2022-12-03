from turtle import left
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import math
# from combined_thresh import combined_thresh
# from perspective_transform import perspective_transform

# feel free to adjust the parameters in the code if necessary

def line_fit(binary_warped):
	"""
	Find and fit lane lines
	"""
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	# Create an output image to draw on and visualize the result
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[0:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 20
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 50
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []
	# create empty lists to receive centroid of left and right boxes
	left_centroids = []
	right_centroids = []
	y_centers = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		
		# get the top and bottom y pixel postions (same for right and left boxed)
		y_bottom = binary_warped.shape[0] - window_height * (window + 1)
		y_top = binary_warped.shape[0] - window_height * (window)
		y_center = (y_top + y_top)/2
		y_centers.append(y_center)

		# get the left and right x pixel positions
		x_left_rect_left = leftx_current - margin
		x_left_rect_right = leftx_current + margin
		x_right_rect_left = rightx_current - margin
		x_right_rect_right = rightx_current + margin

		left_centroids.append(leftx_current)
		right_centroids.append(rightx_current)


		# Draw the windows on the visualization image using cv2.rectangle()
		cv2.rectangle(out_img, (x_left_rect_left, y_top), (x_left_rect_right,y_bottom), color = (255,0,0), thickness=1)
		cv2.rectangle(out_img, (x_right_rect_left, y_top), (x_right_rect_right,y_bottom), color = (255,0,0), thickness=1)
		####
		# Identify the nonzero pixels in x and y within the window
		
		
		left_pixel_indices = [i for i in range(len(nonzerox)) if ((nonzerox[i] >= x_left_rect_left) and (nonzerox[i] < x_left_rect_right) and (nonzeroy[i] >= y_bottom) and (nonzeroy[i] < y_top))]
		right_pixel_indices = [i for i in range(len(nonzerox)) if ((nonzerox[i] >= x_right_rect_left) and (nonzerox[i] < x_right_rect_right) and (nonzeroy[i] >= y_bottom) and (nonzeroy[i] < y_top))]
		
		####
		# Append these indices to the lists
		# create list with all lane midpoints for polyfit
		left_lane_inds.append(np.array(left_pixel_indices, dtype = np.int32))
		right_lane_inds.append(np.array(right_pixel_indices, dtype = np.int32))
		
		####
		# If you found > minpix pixels, recenter next window on their mean position
		##TO DO
	
		if minpix < len(left_pixel_indices):			
			leftx_current = int(np.mean(nonzerox[left_pixel_indices]))			
		if minpix < len(right_pixel_indices):
			rightx_current = int(np.mean(nonzerox[right_pixel_indices]))

		####
		pass

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# cv2.imshow('test', np.float32(out_img))
	# cv2.waitKey(0)

	# Fit a second order polynomial to each using np.polyfit()
	# If there isn't a good fit, meaning any of leftx, lefty, rightx, and righty are empty,
	# the second order polynomial is unable to be sovled.
	# Thus, it is unable to detect edges.
	try:
	##TODO
		# calculate polyfit line for both left and right lanes
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)
	####
	except TypeError:
		print("Unable to detect lanes")
		return None

	x_centers = []
	for i in range(len(left_centroids)):
		temp = (left_centroids[i] + right_centroids[i])/2
		x_centers.append(temp)
        # print(left_cen)
        # print(right_cen)
        # print(y_centers)

	lane_width_meter = 3.03
	lane_width_pixels = 1017
	meter_per_pixel = lane_width_meter/lane_width_pixels

	lane_width_meter_y = 3.18
	lane_width_pixels_y = 720
	meter_per_pixel_y = lane_width_meter_y/lane_width_pixels_y

	# rad_curv_right = np.power(1 + ((2*right_fit[0]*righty[-1]) + right_fit[1])**2, 1.5) / np.abs((2*right_fit[0]))
	# slope = (2*left_fit[0]*lefty[-1]) + left_fit[1]

	# # use first/last waypoint
	# x_2 = (binary_warped.shape[1]/2 - ((rightx[-1] + leftx[-1])/2)) * meter_per_pixel
	# y_2 = (binary_warped.shape[0]- ((righty[-1] + lefty[-1])/2)) * meter_per_pixel

	# x_1 = (binary_warped.shape[1]/2 - ((rightx[0] + leftx[0])/2)) * meter_per_pixel
	# y_1 = (binary_warped.shape[0]- ((righty[0] + lefty[0])/2)) * meter_per_pixel



	if rightx.size > leftx.size:
		# x_2 = (binary_warped.shape[1]/2 - ((rightx[-1] + (rightx[-1] - lane_width_pixels))/2)) * meter_per_pixel
		# y_2 = (binary_warped.shape[0]- ((righty[-1] + righty[-1])/2)) * meter_per_pixel_y

		# x_1 = (binary_warped.shape[1]/2 - ((rightx[0] + (rightx[0] - lane_width_pixels))/2)) * meter_per_pixel
		# y_1 = (binary_warped.shape[0]- ((righty[0] + righty[0])/2)) * meter_per_pixel_y
		print("right biugger")

		x_2 = (binary_warped.shape[1]/2 - ((leftx[-1] + lane_width_pixels + leftx[-1])/2)) * meter_per_pixel
		y_2 = (binary_warped.shape[0]- ((lefty[-1] + lefty[-1])/2)) * meter_per_pixel_y

		x_1 = (binary_warped.shape[1]/2 - ((leftx[0] + lane_width_pixels + leftx[0])/2)) * meter_per_pixel
		y_1 = (binary_warped.shape[0]- ((lefty[0] + lefty[0])/2)) * meter_per_pixel_y
	elif rightx.size < leftx.size:
		x_2 = (binary_warped.shape[1]/2 - ((leftx[-1] + lane_width_pixels + leftx[-1])/2)) * meter_per_pixel
		y_2 = (binary_warped.shape[0]- ((lefty[-1] + lefty[-1])/2)) * meter_per_pixel_y

		x_1 = (binary_warped.shape[1]/2 - ((leftx[0] + lane_width_pixels + leftx[0])/2)) * meter_per_pixel
		y_1 = (binary_warped.shape[0]- ((lefty[0] + lefty[0])/2)) * meter_per_pixel_y
		print("left biugger")

	else:
		x_2 = (binary_warped.shape[1]/2 - ((rightx[-1] + leftx[-1])/2)) * meter_per_pixel
		y_2 = (binary_warped.shape[0]- ((righty[-1] + lefty[-1])/2)) * meter_per_pixel_y

		x_1 = (binary_warped.shape[1]/2 - ((rightx[0] + leftx[0])/2)) * meter_per_pixel
		y_1 = (binary_warped.shape[0]- ((righty[0] + lefty[0])/2)) * meter_per_pixel_y

	cv2.imwrite("test_img.png", out_img)


	# Return a dict of relevant variables
	ret = {}
	ret['left_fit'] = left_fit
	ret['right_fit'] = right_fit
	ret['nonzerox'] = nonzerox
	ret['nonzeroy'] = nonzeroy
	ret['out_img'] = out_img
	# ret['left_lane_inds'] = left_lane_inds
	# ret['right_lane_inds'] = right_lane_inds
	# ret['left_centroids'] = left_centroids
	# ret['right_centroids'] = right_centroids
	# ret['y_centers'] = y_centers
	# print(actual_x_dist)
	# print(actual_y_dist)
	# print(waypoint_heading)
	ret['waypoint_x_1'] = x_1
	ret['waypoint_y_1'] = y_1
	ret['waypoint_x_2'] = x_2
	ret['waypoint_y_2'] = y_2


	return ret

