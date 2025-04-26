import os
import glob

import math
import numpy as np
import cv2

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

import src.features as features
import src.motion as motion



class ImageLoader:

    def __init__(self, project_root_path, folder_KITTI, detector_flag ="sift"):




        # paths
        image_left_dir = os.path.join(project_root_path, folder_KITTI, 'image_l')
        image_right_dir = os.path.join(project_root_path, folder_KITTI, 'image_r')
        poses_path = os.path.join(project_root_path, folder_KITTI, 'poses.txt')
        calibration_path = os.path.join(project_root_path, folder_KITTI, 'calib.txt')

        self.image_paths_left = sorted(os.path.join(image_left_dir, file) for file in sorted(os.listdir(image_left_dir)))
        self.image_paths_right = sorted(os.path.join(image_right_dir, file) for file in sorted(os.listdir(image_right_dir)))



        assert len(self.image_paths_left) == len(self.image_paths_right), "the number of left and right images is not equal"


        self.number_of_images = len(self.image_paths_left)

        print(f"Number of images: {self.number_of_images}")

        self.poses = None
        self.current_image_index = -1


        self.poses_data = np.loadtxt(poses_path)
        #P_l, P_r = data.reshape(-1, 3, 4)  # Each row is a 3x4 projection matrix

        self.K_left, self.P_left, self.K_right, self.P_right = self._load_KITTI_calibration(calibration_path)

        if detector_flag == "sift":
            self.kp_detector = cv2.SIFT_create()


        #block = 11
        #P1 = block * block * 8
        #P2 = block * block * 32
        #self.disparity_estimator = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=block, P1=P1, P2=P2)

        self.not_done = True

        self.prev_image_left        = None
        self.current_image_left     = None
        self.prev_image_right       = None
        self.current_image_right    = None
        self.depth_map_prev         = None
        self.depth_map_current      = None

        self.prev_disparity         = None
        self.current_disparity      = None
        self.kp_prev                = None
        self.kp_current             = None
        self.des_prev               = None
        self.des_current            = None
        self.match                  = None

        self.want_to_filter         = False

        self.current_pose_matrix = np.eye(4)
        self.pos_hom_o = np.array([[0],
                                   [0],
                                   [0],
                                   [1]])

        self.trajectory = []

        self.K_left_inv = np.linalv.inv(self.K_left)


        self.load_next_frame()

    @staticmethod
    def _update_current_prev_numpy(current, new):

        return np.copy(current), new
    def load_next_frame(self):

        self.current_image_index += 1

        new_img_path_left = self.image_paths_left[self.current_image_index]
        new_img_path_right = self.image_paths_right[self.current_image_index]

        new_img_left = cv2.imread(new_img_path_left, cv2.IMREAD_GRAYSCALE)
        new_img_right = cv2.imread(new_img_path_right, cv2.IMREAD_GRAYSCALE)

        (self.prev_image_left,
         self.current_image_left) = self._update_current_prev_numpy(self.current_image_left, new_img_left)

        (self.prev_image_right,
         self.current_image_right) = self._update_current_prev_numpy(self.current_image_right, new_img_right)

        kp_new, des_new = self.detect_current_frame(self)

        (self.kp_prev,
         self.kp_current) = self._update_current_prev_numpy(self.kp_current, kp_new)

        (self.des_prev,
         self.des_current) = self._update_current_prev_numpy(self.des_current, des_new)

        #(self.prev_disparity,
        # self.current_disparity) = self._update_current_prev_numpy(self.current_disparity, self.get_new_disparity())

        #new_depth_map = None



        #(self.depth_map_prev,
        # self.depth_map_current) = self._update_current_prev_numpy(self.depth_map_current, new_depth_map)

        self.not_done = (self.current_image_index < self.number_of_images)

    def show_prev_current_img_disp_map(self):

        images = [self.prev_image_left, self.current_image_left,
                  self.prev_image_right, self.current_image_right] #, self.prev_disparity, self.current_disparity
        titles = ['prev_image_left', 'current_image_left',
                  'prev_image_right', 'current_image_right'] #, 'prev_disparity', 'current_disparity'

        plt.figure(figsize=(15, 8))  # Adjust figure size as needed

        for i, (image, title) in enumerate(zip(images, titles)):
            plt.subplot(2, 3, i + 1)  # 2 rows, 3 columns
            if i<4:
                plt.imshow(image, cmap='gray')
            else:  # Color
                plt.imshow(image, cmap='plasma')  # or 'inferno', 'viridis', etc.
                plt.colorbar(label='Disparity')
            plt.title(title + ' #' + str(self.current_image_index))
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        if False:
            # Show the image
            cv2.imshow(f"self.prev_image_left #{self.current_image_index}", self.prev_image_left)
            cv2.imshow(f"self.current_image_left #{self.current_image_index}", self.current_image_left)
            cv2.imshow(f"self.prev_image_right #{self.current_image_index}", self.prev_image_right)
            cv2.imshow(f"self.current_image_right #{self.current_image_index}", self.current_image_right)
            cv2.imshow(f"self.prev_disparity #{self.current_image_index}", self.prev_disparity)
            cv2.imshow(f"self.current_disparity #{self.current_image_index}", self.current_disparity)
            #cv2.imshow(f"self.depth_map_prev #{self.current_image_index}", self.depth_map_prev)
            #cv2.imshow(f"self.depth_map_current #{self.current_image_index}", self.depth_map_current)

            # Wait for key press (0 = wait indefinitely)
            cv2.waitKey(0)

            # Destroy the window after key press
            cv2.destroyAllWindows()
    def detect_current_frame(self):
        """
        Detects kp, des for the current frame. Only left camera used.
        :return:
        """
        kp_new, des_new = features.extract_features_from_image(self.current_image_left, self.kp_detector)#self.kp_current, self.des_current =
        return kp_new, des_new
    def match_current_frame_with_prev(self):
        self.match = features.match_features(self.des_prev, self.des_current, ratio_test=False)
        if self.want_to_filter:
            self.match = features.filter_by_distance(self.match, 9999999999)

    def estimate_motion_current_frame(self):
        rmat, tvec, image1_points, image2_points = \
            motion.estimate_motion(self.match, self.kp_prev, self.kp_current, self.K_left_inv, depth1=None, extra_output=False)

        return rmat, tvec, image1_points, image2_points

    def estimate_current_position(self):
        rmat, tvec, image1_points, image2_points = self.estimate_motion_current_frame()

        Rt = np.hstack((rmat, tvec))
        Rt = np.vstack((Rt, np.array([0, 0, 0, 1])))
        Ci = np.linalg.inv(Rt)

        self.current_pos = motion.transformation_matrix_to_pos(Ci, self.current_pose_matrix, self.pos_hom_o)
        self.trajectory.append(self.current_pos)
    @staticmethod
    def _load_KITTI_calibration(path):
        """
        Load the camera and projection matrices for the left and right cameras. For the KITTI dataset.
        :param path: path to the file
        :return:
        K_left: intrinsic matrix for the left camera (3x3)
        P_left: projection matrix for the left camera (3x4)
        K_right: intrinsic matrix for the right camera (3x3)
        P_right: projection matrix for the right camera (3x4)
        """


        return K_left, P_left, K_right, P_right

    @staticmethod
    def _load_KITTI_poses(path):
        """
        Loads the ground truth poses from the KITTI dataset
        :param path: Path to poses.txt
        :return: poses, a list of 4x4 [R|t] np.arrays representing the poses
        """

        return poses  # List of 4x4 [R|t] np.arrays



            







