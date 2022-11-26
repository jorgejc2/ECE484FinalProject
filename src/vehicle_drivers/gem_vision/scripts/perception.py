import cv2
import time
import math
import copy
import rospy
import numpy as np

from skimage import morphology
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from gazebo_msgs.srv import GetModelState, GetModelStateResponse

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from Line import Line
from line_fit import line_fit, tune_fit, bird_fit, final_viz

class VehiclePerception:
    def __init__(self, model_name='gem', resolution=0.1, side_range=(-20., 20.),
            fwd_range=(-20., 20.), height_range=(-1.6, 0.5)):
        self.lane_detector = lanenet_detector()
        self.lidar = LidarProcessing(resolution=resolution, side_range=side_range, fwd_range=fwd_range, height_range=height_range)

        self.bridge = CvBridge()
        self.model_name = model_name

        self.node_name = "gem_vision"
        
        rospy.init_node(self.node_name)
        
        rospy.on_shutdown(self.cleanup)

    def cleanup(self):
        print ("Shutting down vision node.")
        cv2.destroyAllWindows()


    def cameraReading(self):
        # Get processed reading from the camera on the vehicle
        # Input: None
        # Output:
        # 1. Lateral tracking error from the center line of the lane
        # 2. The lane heading with respect to the vehicle
        return self.lane_detector.lateral_error, self.lane_detector.lane_theta

    def lidarReading(self):
        # Get processed reading from the Lidar on the vehicle
        # Input: None
        # Output: Distance between the vehicle and object in the front
        res = self.lidar.processLidar()
        return res

    def gpsReading(self):
        # Get the current state of the vehicle
        # Input: None
        # Output: ModelState, the state of the vehicle, contain the
        #   position, orientation, linear velocity, angular velocity
        #   of the vehicle
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            modelState = serviceResponse(model_name=self.model_name)
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
            modelState = GetModelStateResponse()
            modelState.success = False
        return modelState

class LidarProcessing:
    def __init__(self, resolution=0.1, side_range=(-20., 20.), fwd_range=(-20., 20.),
                         height_range=(-1.6, 0.5)):
        self.resolution = resolution
        self.side_range = side_range
        self.fwd_range = fwd_range
        self.height_range = height_range

        self.cvBridge = CvBridge()

        # empty initial image
        self.birdsEyeViewPub = rospy.Publisher("/mp4/BirdsEye", Image, queue_size=1)
        self.pointCloudSub = rospy.Subscriber("/velodyne_points", PointCloud2, self.__pointCloudHandler, queue_size=10)
        x_img = np.floor(-0 / self.resolution).astype(np.int32)
        self.vehicle_x = x_img - int(np.floor(self.side_range[0] / self.resolution))

        y_img = np.floor(-0 / self.resolution).astype(np.int32)
        self.vehicle_y = y_img + int(np.ceil(self.fwd_range[1] / self.resolution))


        self.x_front = float('nan')
        self.y_front = float('nan')

    def __pointCloudHandler(self, data):
        """
            Callback function for whenever the lidar point clouds are detected

            Input: data - lidar point cloud

            Output: None

            Side Effects: updates the birds eye view image
        """
        gen = point_cloud2.readgen = point_cloud2.read_points(cloud=data, field_names=('x', 'y', 'z', 'ring'))

        lidarPtBV = []
        for p in gen:
            lidarPtBV.append((p[0],p[1],p[2]))

        self.construct_birds_eye_view(lidarPtBV)

    def construct_birds_eye_view(self, data):
        """
            Call back function that get the distance between vehicle and nearest wall in given direction
            The calculated values are stored in the class member variables

            Input: data - lidar point cloud
        """
        # create image from_array
        x_max = 1 + int((self.side_range[1] - self.side_range[0]) / self.resolution)
        y_max = 1 + int((self.fwd_range[1] - self.fwd_range[0]) / self.resolution)
        im = np.zeros([y_max, x_max], dtype=np.uint8)

        if len(data) == 0:
            return im

        # Reference: http://ronny.rest/tutorials/module/pointclouds_01/point_cloud_birdseye/
        data = np.array(data)

        x_points = data[:, 0]
        y_points = data[:, 1]
        z_points = data[:, 2]

        # Only keep points in the range specified above
        x_filter = np.logical_and((x_points >= self.fwd_range[0]), (x_points <= self.fwd_range[1]))
        y_filter = np.logical_and((y_points >= self.side_range[0]), (y_points <= self.side_range[1]))
        z_filter = np.logical_and((z_points >= self.height_range[0]), (z_points <= self.height_range[1]))

        filter = np.logical_and(x_filter, y_filter)
        filter = np.logical_and(filter, z_filter)
        indices = np.argwhere(filter).flatten()

        x_points = x_points[indices]
        y_points = y_points[indices]
        z_points = z_points[indices]

        def scale_to_255(a, min_val, max_val, dtype=np.uint8):
            a = (((a-min_val) / float(max_val - min_val) ) * 255).astype(dtype)
            tmp = copy.deepcopy(a)
            a[:] = 0
            a[tmp>0] = 255
            return a

        # clip based on height for pixel Values
        pixel_vals = np.clip(a=z_points, a_min=self.height_range[0], a_max=self.height_range[1])

        pixel_vals = scale_to_255(pixel_vals, min_val=self.height_range[0], max_val=self.height_range[1])

        # Getting sensor reading for front
        filter_front = np.logical_and((y_points>-2), (y_points<2))
        filter_front = np.logical_and(filter_front, x_points > 0)
        filter_front = np.logical_and(filter_front, pixel_vals > 128)
        indices = np.argwhere(filter_front).flatten()

        self.x_front = np.mean(x_points[indices])
        self.y_front = np.mean(y_points[indices])

        # convert points to image coords with resolution
        x_img = np.floor(-y_points / self.resolution).astype(np.int32)
        y_img = np.floor(-x_points / self.resolution).astype(np.int32)

        # shift coords to new original
        x_img -= int(np.floor(self.side_range[0] / self.resolution))
        y_img += int(np.ceil(self.fwd_range[1] / self.resolution))

        # Generate a visualization for the perception result
        im[y_img, x_img] = pixel_vals

        img = im.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        center = (self.vehicle_x, self.vehicle_y)
        cv2.circle(img, center, 5, (0,0,255), -1, 8, 0)

        center = self.convert_to_image(self.x_front, self.y_front)
        cv2.circle(img, center, 5, (0,255,0), -1, 8, 0)
        if not np.isnan(self.x_front) and not np.isnan(self.y_front):
            cv2.arrowedLine(img, (self.vehicle_x,self.vehicle_y), center, (255,0,0))

        x1, y1 = self.convert_to_image(20,2)
        x2, y2 = self.convert_to_image(0,-2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0))

        birds_eye_im = self.cvBridge.cv2_to_imgmsg(img, 'bgr8')

        self.birdsEyeViewPub.publish(birds_eye_im)


    def convert_to_image(self, x, y):
        """
            Convert point in vehicle frame to position in image frame
            Inputs:
                x: float, the x position of point in vehicle frame
                y: float, the y position of point in vehicle frame
            Outputs: Float, the x y position of point in image frame
        """

        x_img = np.floor(-y / self.resolution).astype(np.int32)
        y_img = np.floor(-x / self.resolution).astype(np.int32)

        x_img -= int(np.floor(self.side_range[0] / self.resolution))
        y_img += int(np.ceil(self.fwd_range[1] / self.resolution))
        return (x_img, y_img)

    def processLidar(self):
        """
            Compute the distance between vehicle and object in the front
            Inputs: None
            Outputs: Float, distance between vehicle and object in the front
        """
        front = np.sqrt(self.x_front**2+self.y_front**2)

        return front

    def get_lidar_reading(self):
        pass

class lanenet_detector():
    def __init__(self):

        self.bridge = CvBridge()
        # self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
        # self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        # self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False

        # initialization for the lateral tracking error and lane heading
        self.lateral_error = 0.0
        self.lane_theta = 0.0

        # determine the meter-to-pixel ratio
        lane_width_meters = 4.4
        lane_width_pixels = 265.634
        self.meter_per_pixel = lane_width_meters / lane_width_pixels


    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        mask_image, bird_image, lateral_error, lane_theta = self.detection(raw_img)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)

        # publish the lateral tracking error and lane heading
        if lateral_error is not None:
            self.lateral_error = lateral_error

        if lane_theta is not None:
            self.lane_theta = lane_theta


    def gradient_thresh(self, img, thresh_min=25, thresh_max=100):
        """
        Apply sobel edge detection on input image in x, y direction
        """
        #1. Convert the image to gray scale
        #2. Gaussian blur the image
        #3. Use cv2.Sobel() to find derievatives for both X and Y Axis
        #4. Use cv2.addWeighted() to combine the results
        #5. Convert each pixel to unint8, then apply threshold to get binary image

        ## TODO
        # convert img to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # gaussian blur
        cv2.GaussianBlur(gray_img, (5, 5), cv2.BORDER_DEFAULT)

        # use sobel filter
        grad_x = cv2.Sobel(gray_img, cv2.CV_16S, 1, 0, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray_img, cv2.CV_16S, 0, 1, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)


        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        # combine results
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        # convert pixels to uint8 and apply threshold
        thresh, binary_output = cv2.threshold(grad, thresh_min, thresh_max, cv2.THRESH_BINARY)
        binary_output[(binary_output > 0)] = 1
        ####

        return binary_output



    def color_thresh(self, img, thresh=(100, 255)):
        """
        Convert RGB to HSL and threshold to binary image using S channel
        """
        #1. Convert the image from RGB to HSL
        #2. Apply threshold on S channel to get binary image
        #Hint: threshold on H to remove green grass
        ## TODO

        ####
        # convert to HLS
        hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hls_img = np.uint8(hls_img)


        # apply threshold on S channel
        binary_output = np.copy(hls_img)

        binary_output[:,:,:] = 0
        binary_output[(hls_img[:,:,1] > 250)] = 1

        binary_output = cv2.cvtColor(cv2.cvtColor(binary_output, cv2.COLOR_HLS2RGB), cv2.COLOR_RGB2GRAY)

        return binary_output


    def combinedBinaryImage(self, img):
        """
        Get combined binary image from color filter and sobel filter
        """
        # TODO: Use your MP1 implementation for this function
        #1. Apply sobel filter and color filter on input image
        #2. Combine the outputs
        ## Here you can use as many methods as you want.

        # ## TODO
        # SobelOutput = self.gradient_thresh(img, 15, 255)
        # ColorOutput = self.color_thresh(img, (20, 90))
        # ####

        # binaryImage = np.zeros_like(SobelOutput)
        # binaryImage[(ColorOutput==1)|(SobelOutput==1)] = 255

      
        # # Remove noise from binary image
        # binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2)

        # binaryImage = CombineGradientColor(img, s_thresh=(170,255), sx_thresh=(20,100))
        binaryImage = Gradient(img, threshold=(60,100))
        return binaryImage



    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """
        #1. Visually determine 4 source points and 4 destination points
        #2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        #3. Generate warped image in bird view using cv2.warpPerspective()

        ## TODO
        # # GEM simulation:
        # source = np.float32([[0,440],[238,290],[400,290],[640,440]])
        # destination = np.float32([[80,410],[80,70],[560,70],[560,410]])
        # source = np.float32([[40,440],[280,300],[360,300],[600,440]])
        # destination = np.float32([[80,440],[80,70],[560,70],[560,440]])

        # original points from mp1
        # source = np.float32([[0,420],[220,290],[410,290],[640,420]])
        # destination = np.float32([[0,480],[0,0],[640,0],[640,480]])
        # points for the GEM car
        source = np.float32([[300,680],[440,500],[810,500],[980,680]])
        destination = np.float32([[50,700],[50,20],[1230,20],[1230,700]])
        # points for the GEM car
        # source = np.float32([[300,680],[450,480],[830,480],[980,680]])
        # destination = np.float32([[0,720],[0,0],[1280,0],[1280,720]])

        # ROSBag Simulation
        # source = np.float32([[250,370],[420,250],[800,250],[950,370]])
        # destination = np.float32([[0,375],[0,0],[1242,0],[1242,375]])
        # print(img.shape)

        M = cv2.getPerspectiveTransform(source, destination)
        Minv = np.linalg.inv(M)

        img_size = (img.shape[1], img.shape[0])
        warped_img = cv2.warpPerspective(np.float32(img), M, dsize=img_size)
        # warped_img = cv2.warpPerspective(np.float32(img), M, dsize=(1242,375))
        ####

        return warped_img, M, Minv



    def detection(self, img):

        binary_img = self.combinedBinaryImage(img)
        img_birdeye, M, Minv = self.perspective_transform(binary_img)

        center_x = int(img_birdeye.shape[1] / 2)
        height = img_birdeye.shape[0]

        # Fit lane with the newest image
        ret = line_fit(img_birdeye)

        if ret is not None:
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']

            left_centroids = ret['left_centroids']
            right_centroids = ret['right_centroids']
            y_centers = ret['y_centers']

            left_fit = self.left_line.add_fit(left_fit)
            right_fit = self.right_line.add_fit(right_fit)

        # return lane detection results
        bird_fit_img = None
        combine_fit_img = None
        lateral_error = None
        lane_theta = None
        if ret is not None:
            bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
            combine_fit_img = final_viz(img, left_fit, right_fit, Minv)

            # instead of estimating the lateral tracking error and the lane heading separately at
            # different locations in the bird's eye view image as shown below, one can set a point
            # along the center line in front of the vehicle as a reference point and use a similar
            # controller as in MP2.

            # TODO :calculate the lateral tracking error from the center line
            # Hint: positive error should occur when the vehicle is to the right of the center line

            left_bottom = left_centroids[0]
            right_bottom = right_centroids[0]

            lane_center_x = (left_bottom + right_bottom)/2

            lateral_error = center_x - lane_center_x
            lateral_error *= self.meter_per_pixel


            # TODO: calculate the lane heading error
            # Hint: use the lane heading a few meters before the vehicle to avoid oscillation
            
            # left_ref_heading = left_centroids[4]
            # right_ref_heading = right_centroids[4]
            input_y = y_centers[4]

            # if you need to turn left, you have a positive heading error
            print("len left indices: {}".format(len(left_lane_inds)))
            print("len right indices: {}".format(len(right_lane_inds)))

            slope_left = None
            slope_right = None

            if left_fit is None or len(left_lane_inds) < 5000:
                slope_right = 2*right_fit[0]*input_y + right_fit[1]
                heading_right = np.arctan(-slope_right)
                lane_theta = heading_right
                print(f"right fit coefficients: {right_fit}")
            elif right_fit is None or len(right_lane_inds) < 5000:
                slope_left = 2*left_fit[0]*input_y + left_fit[1]
                heading_left = np.arctan(-slope_left)
                lane_theta = heading_left
                print(f"left fit coefficients: {left_fit}")
            else:
                slope_left = 2*left_fit[0]*input_y + left_fit[1]
                slope_right = 2*right_fit[0]*input_y + right_fit[1]
                heading_left = np.arctan(-slope_left)
                heading_right = np.arctan(-slope_right)
                lane_theta = (heading_left + heading_right) / 2
                print(f"left fit coefficients: {left_fit}")
                print(f"right fit coefficients: {right_fit}")


            # print(f"input_y: {input_y}")
            print(f"slope_left: {slope_left}")
            print(f"slope_right: {slope_right}")
            print(f"lateral error: {lateral_error}")
            print(f"lane_theta: {lane_theta}")
            print("---------------\n")

            # equation for controller
            # new heading = curr_heading + 4pi(heading_error/max_steering_angle) + 4pi*arctan((max_lateral_error - lateral_error) / lateral_error)

            
        else:
            print("Unable to detect lanes")

        return combine_fit_img, bird_fit_img, lateral_error, lane_theta

def Color(img, threshold = (90,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    binary = np.zeros_like(S)
    binary[(S > threshold[0]) & (S <= threshold[1])] = 1
    return binary


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # Return the result
    return binary_output


# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def Gradient(img, threshold = (0,255)):
    # Choose a Sobel kernel size
    ksize = 5  # Choose a larger odd number to smooth gradient measurements
    image = img

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=threshold)
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=threshold)
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=threshold)
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(-np.pi / 2, np.pi / 2))
    combined = np.zeros_like(dir_binary)
    # combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined[(gradx == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined


def CombineGradientColor(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    # Note: img is the undistorted image
    # img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
     # Grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary


def visualize(img):
    """Plot the images for README"""
    # image = mpimg.imread('./test_images/test5.jpg')
    image = img
    # CombineColorGradient
    f, axes = plt.subplots(2, 2, figsize=(10, 6))
    (ax1, ax2, ax3, ax4) = axes.ravel()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=12)
    ax2.imshow(Color(image, threshold=(170,255)), cmap='gray')
    cv2.imshow('test', np.float32(Color(image, threshold = (170, 255))))
    cv2.waitKey(0)
    ax2.set_title('S threshold', fontsize=12)
    ax3.imshow(Gradient(image, threshold=(60,100)), cmap='gray')
    cv2.imshow('test', np.float32(Gradient(image, threshold=(60,100))))
    cv2.waitKey(0)
    ax3.set_title('Sobel x, gradient threshold', fontsize=12)
    ax4.imshow(CombineGradientColor(image, s_thresh=(170,255), sx_thresh=(20,100)),
               cmap='gray')
    cv2.imshow('test', np.float32(CombineGradientColor(image, s_thresh=(170,255), sx_thresh=(20,100))))
    cv2.waitKey(0)
    ax4.set_title('Combined S Channel & gradient threshold', fontsize=12)
    for ax in axes.ravel():
        ax.axis('off')




if __name__ == "__main__":
    # VehiclePerception('gem1')

    # rospy.spin()
    img = mpimg.imread('./raw_image.jpg')
    visualize(img)
