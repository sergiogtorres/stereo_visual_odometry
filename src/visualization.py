import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
import os



def show_reprojection_mismatch(object_points_prev, image_points_current, rvec, tvec, k, current_image_index):
    """
    This function only checks that the rvec, tvec transformation of object_points (3D) from prev -> image_points (2D)
    current image
    into
    :param object_points_prev:
    :param image_points_current:
    :param rvec:
    :param tvec:
    :param k:
    :param current_image_index:
    :return:
    """
    projected_points, _ = cv2.projectPoints(object_points_prev, rvec, tvec, k, None)
    projected_points = projected_points.squeeze()

    plt.figure("reprojection error" + str(current_image_index))
    plt.scatter(image_points_current[:, 0], image_points_current[:, 1], c='r', label="Image Points")
    plt.scatter(projected_points[:, 0], projected_points[:, 1], c='black', s=5, label="Projected Points")
    plt.legend()
    plt.show()



""" ^^^^^^^^^^^^^^^^^^^
(rmat, tvec, image1_points,
 image2_points, success,
 inliers, object_points,
 image_points, rvec) = estimate_motion(match, kp1, kp2, k, depth1=depth, extra_output=True)"""




def trajectory_to_xyz(trajectory):
    matrix = np.array([[1, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, -1, 0, 0],
                       [0, 0, 0, 1]])
    out = matrix @ trajectory
    return out


def visualize_trajectory(trajectory):
    """

    :param trajectory: 3xN array    [[x, x, x...],
                                     [y, y, y... ],
                                     [z, z, z... ]
    :return:
    """
    fig_3d = plt.figure("estimated trajectory 3d")
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    # Extract x, y, z
    x, y, z = trajectory[0], trajectory[1], trajectory[2]

    # Plot trajectory
    ax_3d.plot(x, y, z, label='Camera Trajectory', color='blue')
    ax_3d.scatter(x[0], y[0], z[0], color='green', label='Start')
    ax_3d.scatter(x[-1], y[-1], z[-1], color='red', label='End')

    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.legend()

    # Equal aspect ratio
    max_range = np.ptp([x, y, z]).max() / 2.0
    mid_x = np.mean(x)
    mid_y = np.mean(y)
    mid_z = np.mean(z)

    ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)

    fig_2d = plt.figure("estimated trajectory 2d")
    ax_2d = fig_2d.add_subplot(111)

    ax_2d.plot(x, y, label='Camera Trajectory', color='blue')
    ax_2d.scatter(x[0], y[0], color='green', label='Start')
    ax_2d.scatter(x[-1], y[-1], color='red', label='End')

    ax_2d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_2d.set_ylim(mid_y - max_range, mid_y + max_range)

    plt.show(block=False)
    plt.pause(0.001)


def make_output_video(images_out_path, file_header):
    in_file_path = f"{images_out_path}/{file_header}"
    filenames = sorted(glob.glob(f"{in_file_path}_*.png"))

    video_out_path = f"{images_out_path}/video"
    video_out_filepath = f'{video_out_path}/animation.mp4'
    frame = cv2.imread(filenames[0])
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(video_out_path, exist_ok=True)
    out = cv2.VideoWriter(video_out_filepath, fourcc, 5.0, (width, height))
    for fname in filenames:
        img = cv2.imread(fname)
        out.write(img)

    print(f'saving at: {video_out_filepath}')
    out.release()