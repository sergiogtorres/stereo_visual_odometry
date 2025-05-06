import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
from src.utils import *
from src.features import *
from src.motion import *
import src.debug_utils
import sys

import glob
matplotlib.use('TkAgg')  # Or 'Qt5Agg', 'WebAgg', etc.

# TODO: use optical flow for most frames, and keypoint matching for keyframes (e.g. Kalman Filter update step)
# TODO: with optical flow tracking of features, sparse triangulation (instead of depth maps) makes more sense
#  (
#  without optical flow, there would need to be 2 matching steps per frame:
#  current_left <-> current_right
#  current_left <-> previous_left
#  (triangulation requires left&right views, +1 view to get the [R|t] transformation from previous_left to current_left)
#  )
if __name__ == "__main__":

    trajectory = []
    root_dir_path = os.path.dirname(os.path.realpath(__file__))#'F:\\projects\\1.start_projects_here\\stereo_visual_odometry'
    folder_KITTI = os.path.join(root_dir_path, "data/KITTI_sequence_10")
    print(root_dir_path)
    #if 'image_handler' not in globals():
    #    image_handler = ImageHandler(root_dir_path)
    image_handler = ImageHandler(root_dir_path, folder_KITTI, create_plots=False)

    iteration = 0
    #k = image_handler.k
    while image_handler.not_done:
        print(f"iteration:{iteration}")
        #image_handler.print_types_current_and_prev_frames()
        image_handler.load_next_frame() # loads the next set of images and calculates the depth map
        # ^^ TODO: Should start at image 1 as current, then with load_next_frame() load the second one as current
        # I. Feature Extraction
        print("extracting features")
        #image_handler.detect_current_frame() #at all times, we are comparing previous with current


        # II. Feature Matching

        image_handler.match_prev_frame_with_current()
        if image_handler.create_plots:
            image_handler.show_current_and_prev_frames()
            show_matches(image_handler.prev_image_left,
                         image_handler.kp_prev,
                         image_handler.current_image_left,
                         image_handler.kp_current,
                         image_handler.match,
                         image_handler.current_image_index,
                         image_handler)


        #matches = image_handler.current_matches #match_features_dataset(des_list, match_features)

        # Part III. Trajectory Estimation

        #image_handler.estimate_frame_motion()
        image_handler.estimate_current_position() #calculates the current point by applying transformation Ci matrix
        trajectory.append(image_handler.current_pos[:3])


        iteration += 1


    if image_handler.create_plots:
        visualization.make_output_video(image_handler.output_video_path, image_handler.frame_name)
        visualization.make_output_video(image_handler.output_video_path_matches, image_handler.frame_name_matches)

    trajectory = np.array(trajectory)#estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=depth_maps, debug=False)

    #trajectory_x_right_y_front_z_up = visualization.trajectory_to_xyz(trajectory)
    #visualization.visualize_trajectory(trajectory_x_right_y_front_z_up)

    plt.figure('x-z estimated and GT')
    plt.axis('equal')
    plt.plot(trajectory[:, 0], trajectory[:, 2])

    poses_GT, _ = image_handler._load_KITTI_poses(image_handler.poses_path)
    poses_manual = []
    for element_ in poses_GT:
        xyz = element_[:, -1]
        poses_manual.append(xyz)
    poses_manual = np.array(poses_manual)
    plt.plot(poses_manual[:, 0], poses_manual[:, 2], color='red')

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


