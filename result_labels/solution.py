import cv2
import numpy as np
import os

focal_length = 910
width = 1164
height = 874
center_x = width/2
center_y = height/2
pp = (center_x, center_y)

sift = cv2.SIFT_create()

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass an empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)

K = np.array([[focal_length, 0, center_x],
              [0, focal_length, center_y],
              [0, 0, 1]])

def read_specific_frame(video_directory, frame_number):
    # Construct the path to the specific frame
    frame_path = os.path.join("/Users/ritamsaha/Desktop/whereabouts/calib_challenge/result_labels",video_directory, f"{frame_number}.jpg")
    print(frame_path)

    # Check if the frame exists
    if not os.path.exists(frame_path):
        print(f"Frame {frame_number} does not exist in the directory.")
        return None

    # Read the frame
    frame = cv2.imread(frame_path)

    return frame

def extract_sift_features(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return keypoints, descriptors

def match_features(des1, kp1, des2, kp2):
    # Match descriptors and apply Lowe's ratio test
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Extract location of good matches
    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros_like(points1)

    for i, match in enumerate(good_matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Find homography using RANSAC
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use mask to select the inlier matches
    inlier_matches = [good_matches[i] for i in range(len(mask)) if mask[i]]

    return inlier_matches

frame1 = read_specific_frame("0",10)
frame2 = read_specific_frame("0",10)

kp1, des1 = extract_sift_features(frame1)
kp2, des2 = extract_sift_features(frame2)

matches = match_features(des1, kp1, des2, kp2)

points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# Calculate Essential Matrix
F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
E, mask = cv2.findEssentialMat(points1, points2, K)

_, R, t, mask = cv2.recoverPose(E, points1, points2, focal=focal_length, pp=pp)


print("Essential Matrix:\n", E)

print("Translation Vector:\n", t)
