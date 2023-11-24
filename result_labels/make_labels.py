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

def draw_matches(img1, kp1, img2, kp2, matches, mask):
    # Convert mask to a list of booleans
    matchesMask = mask.ravel().tolist()

    # Draw matches
    draw_params = dict(matchColor=(0, 255, 0),  # Draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # Draw only inliers
                       flags=2)

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
    return img_matches

def extract_sift_features(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return keypoints, descriptors

def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def get_yaw_and_pitch(frameNumber):
    frame1 = read_specific_frame("0", frameNumber)
    frame2 = read_specific_frame("0", frameNumber+1)

    kp1, des1 = extract_sift_features(frame1)
    kp2, des2 = extract_sift_features(frame2)

    matches = match_features(des1, kp1, des2, kp2)

    points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(points1, points2, K)

    _, R, t, mask = cv2.recoverPose(E, points1, points2, focal=focal_length, pp=pp)

    yaw, pitch, roll = rotationMatrixToEulerAngles(R)


    return pitch, yaw, roll


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

# for i in range(10):
#     yaw, pitch, roll = get_yaw_and_pitch(i)
#     print("Pitch (Y-axis rotation):", '{:.2e}'.format(np.degrees(pitch)), "degrees")
#     print("Yaw (Z-axis rotation):", '{:.2e}'.format(np.degrees(yaw)), "degrees")
#     print("\n")


frame1 = read_specific_frame("0",0)
frame2 = read_specific_frame("0",1)

kp1, des1 = extract_sift_features(frame1)
kp2, des2 = extract_sift_features(frame2)

matches = match_features(des1, kp1, des2, kp2)

points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# Calculate Essential Matrix
F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
E, mask = cv2.findEssentialMat(points1, points2, K)

print("Essential Matrix:\n", E)

pp = (center_x, center_y)  # your camera's principal point

# Recover the pose from the essential matrix
_, R, t, mask = cv2.recoverPose(E, points1, points2, focal=focal_length, pp=pp)

print("Rotation Matrix:\n", R)
print("Translation Vector:\n", t)

# yaw, pitch, roll = get_yaw_and_pitch(0)

# print("Pitch (Y-axis rotation):", '{:.2e}'.format(np.degrees(pitch)), "degrees")
# print("Yaw (Z-axis rotation):", '{:.2e}'.format(np.degrees(yaw)), "degrees")
# print("Roll (X-axis rotation):", '{:.2e}'.format(np.degrees(roll)), "degrees")

U, S, Vt = np.linalg.svd(F)

# The epipoles are the last column of V (for the first image) and the last row of U (for the second image)
e1 = Vt[-1]
e2 = U[:, -1]

# Normalize the epipoles to get image coordinates
e1 = e1/e1[-1]
e2 = e2/e2[-1]

print(f"Epipole in first image: {e1}")
print(f"Epipole in second image: {e2}")

# epipole_radius = 5
# epipole_color = (0, 0, 255)  # Red color in BGR
# epipole_thickness = 2

# # Draw the epipole on the first frame
# epipole1 = (int(e1[0]), int(e1[1]))
# cv2.circle(frame1, epipole1, epipole_radius, epipole_color, epipole_thickness)

# # Draw the epipole on the second frame
# epipole2 = (int(e2[0]), int(e2[1]))
# cv2.circle(frame2, epipole2, epipole_radius, epipole_color, epipole_thickness)

# cv2.imshow("Frame 1 with Epipole", frame1)
# cv2.imshow("Frame 2 with Epipole", frame2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # img_matches = draw_matches(frame1, kp1, frame2, kp2, matches, mask)
# cv2.imshow("Matched Features", img_matches)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Release the video capture object and close the windows
#cv2.destroyAllWindows()


