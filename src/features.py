import numpy as np
import cv2
from matplotlib import pyplot as plt
import os



def extract_features_from_image(image, detector):
    """
    Extract keypoints and descriptors from an image using an imput kp_detector.
    :param image: input image
    :param detector: kp_detector to use
    :return: kp, des, the lists of detectors and keypoints
    """

    kp, des = detector.detectAndCompute(image, None)


    return kp, des

def show_features(image, kp):
    """
    Visualizes the features in the image
    :param image: image to display on
    :param kp: keypoints from the image
    :return:
    """
    display = cv2.drawKeypoints(image, kp, None)
    plt.imshow(display)

    return


def match_features(des1, des2, ratio_test=False):
    """
    Matches features from two images
    :param des1: list of descriptors from the first image
    :param des2: list of descriptors from the second image
    :param ratio_test:
    :return: match: list of matched features
    TODO: take matcher definition outside
    """

    # Initialize FLANN-based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find matches
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    ratio_thresh = 0.4  # Lower is more selective!!!!
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    matches = good_matches

    # Sort matches by distance
    match = sorted(matches, key=lambda x: x.distance)


    return match


# Optional
def filter_by_distance(match, dist_threshold):
    """
    Filters the matched features according to a distance treshold
    :param match: list of matched features
    :param dist_threshold: maximum normalized distance between best matches in the range [0, 1]
    :return:
    """

    filtered_match = []

    for m in match:
        # print (m.distance)
        # print("m.distance: ", m.distance)
        if m.distance <= dist_threshold:
            filtered_match.append(m)


    return filtered_match

def show_matches(image1, kp1, image2, kp2, match, current_image_index, image_handler):
    """
    Visualizes the matches bwtween two images
    :param image1: first image
    :param kp1: list of first image's keypoints
    :param image2: second image
    :param kp2: list of second image's keypoints
    :param match: list of matched features from the first and second images
    :return:
    """

    image_matches = cv2.drawMatches(image1, kp1, image2, kp2, match,None)
    plt.figure("matches"+str(current_image_index))
    plt.clf()  # Clear previous contents, if any
    plt.imshow(image_matches)

    if image_handler.save_out_video:
        os.makedirs(image_handler.output_video_path_matches, exist_ok=True)
        plt.savefig(f'{image_handler.output_video_path_matches}/'
                    f'{image_handler.frame_name_matches}_{image_handler.current_image_index:04d}.png', dpi=200)
        plt.close()

    return




