import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
from src.utils import *
from src.features import *
from src.motion import *
import sys
matplotlib.use('TkAgg')  # Or 'Qt5Agg', 'WebAgg', etc.

if __name__ == "__main__":

    trajectory = []
    root_dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_KITTI = os.path.join(root_dir_path, "data/KITTI_sequence_10")
    print(root_dir_path)
    #if 'image_handler' not in globals():
    #    image_handler = ImageHandler(root_dir_path)
    image_handler = ImageHandler(root_dir_path, folder_KITTI)


    #k = image_handler.k
    while image_handler.not_done:
        image_handler.load_next_frame() # loads the next set of images and calculates the depth map
        # ^^ TODO: Should start at image 1 as current, then with load_next_frame() load the second one as current
        # I. Feature Extraction
        print("extracting features")
        image_handler.detect_current_frame() #at all times, we are comparing current with previous


        # II. Feature Matching

        image_handler.match_current_frame_with_prev()

        matches = image_handler.current_matches #match_features_dataset(des_list, match_features)

        # Part III. Trajectory Estimation

        image_handler.estimate_frame_motion()
        image_handler.estimate_current_position() #calculates the current point by applying transformation Ci matrix
        trajectory.append(image_handler.current_pos[:3])


    trajectory = np.array(trajectory)#estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=depth_maps, debug=False)
    trajectory_x_right_y_front_z_up = trajectory_to_xyz(trajectory)
    visualize_trajectory(trajectory_x_right_y_front_z_up)

    # images = image_handler.images
    # depth_maps = image_handler.depth_maps
    # kp_list, des_list = extract_features_dataset(images, extract_features)
    # depth_maps = image_handler.depth_maps
        # Set to True if you want to use filtered matches or False otherwise
        #is_main_filtered_m = False
        #if is_main_filtered_m:
        #    dist_threshold = 0.75
        #    filtered_matches = filter_matches_dataset(filter_matches_distance, matches, dist_threshold)
        #    matches = filtered_matches
