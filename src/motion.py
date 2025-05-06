import numpy as np
import cv2
from matplotlib import pyplot as plt
import seaborn
def estimate_motion(match, kp1, kp2, k, k_inv, depth1=None, extra_output=False):
    """
    TODO: refactor from using depth map into using cv2.triangulatePoints()
    Estimates camera motion from the first to the second frame

    :param match: list of features matched from two frames
    :param kp1: list of keypoints from the first image
    :param kp2: list of keypoints from the second image
    :param k_inv: the inverse of the camera intrinsics
    :param depth1: depth map of the first frame. Optional depending on chosen method.
    :param extra_output: boolean flag for debugging
    :return:
    """

    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    image1_points = []
    image2_points = []

    N = len(match)
    object_points = []
    image_points = []
    #k_inv = np.linalg.inv(k)
    for i, mm in enumerate(match):
        # print(i)
        keypoint1 = kp1[mm.queryIdx]
        keypoint2 = kp2[mm.trainIdx]
        image1_points.append(keypoint1.pt)
        image2_points.append(keypoint2.pt)  # (x1+10, y1)

        x1_subpix, y1_subpix = keypoint1.pt
        y1 = int(y1_subpix)
        x1 = int(x1_subpix)

        x2_subpix, y2_subpix = keypoint2.pt
        y2 = int(y2_subpix)
        x2 = int(x2_subpix)

        if depth1 is not None:
            zw1 = depth1[y1, x1]#*np.random.rand()#*20
            if zw1 < 1000:
                pos_feature_cam = np.array([[x1],
                                    [y1],
                                    [1]])

                pos_feature_world = k_inv @ pos_feature_cam
                pos_feature_world /= pos_feature_world[-1, 0]
                pos_feature_world *= zw1
                object_point = pos_feature_world[:, 0]      #previous frame
                image_point = np.array([x2, y2])    #current frame

                object_points.append(object_point)
                image_points.append(image_point)

        else:
            print("not implemented")
            #object_point = cv2.triangulatePoints(P1, P2, image1_points, image2_points)

            #image_point = np.array([x2, y2])    #current frame

            #object_points.append(object_point)
            #image_points.append(image_point)not implemented

    object_points = np.array(object_points)
    image_points = np.array(image_points, dtype=float)
    print(len(object_points), len(image_points))
    success, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, k, None)
    rmat, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to matrix


    if extra_output:
        out = rmat, tvec, image1_points, image2_points, success, inliers, object_points, image_points, rvec #image_points: ignore
    else:
        out = rmat, tvec, image1_points, image2_points
    return out



def transformation_matrix_to_pos(Ci, matrix_prev, pos_hom_o=np.array([[0],[0],[0],[1]])):
    """
    Takes the cummulative transformation matrix Ci and extracts the current pose with respect to the original point,
    in the camera's frame of reference.
    :param Ci:
    :param matrix_prev:
    :param pos_hom_o:
    :return:
    """

    print(f"Ci:\n{Ci}")
    new_matrix = matrix_prev @ Ci

    pos_hom_i = (new_matrix @ pos_hom_o)[:, 0]

    return new_matrix, pos_hom_i
