import time
import math
import numpy as np
import cv2
import rospy

from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology



class lanenet_detector():
    def __init__(self):

        self.bridge = CvBridge()
        # NOTE
        # Uncomment this line for lane detection of GEM car in Gazebo
        # self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
        # Uncomment this line for lane detection of videos in rosbag   
        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True


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
        #1. Apply sobel filter and color filter on input image
        #2. Combine the outputs
        ## Here you can use as many methods as you want.

        ## TODO
        SobelOutput = self.gradient_thresh(img, 15, 255)
        ColorOutput = self.color_thresh(img, (20, 90))
        ####

        binaryImage = np.zeros_like(SobelOutput)
        binaryImage[(ColorOutput==1)|(SobelOutput==1)] = 255

      
        # Remove noise from binary image
        binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2)

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
        source = np.float32([[0,420],[220,290],[410,290],[640,420]])
        destination = np.float32([[0,480],[0,0],[640,0],[640,480]])

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
