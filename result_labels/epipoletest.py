import cv2
import numpy as np
import os

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

    # Create a SIFT object
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return keypoints, descriptors

def match_features(des1, des2):
    # Create BFMatcher object
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass an empty dictionary

    # Create FLANN-based matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

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

width = 4032
height = 3024
focal_length = 319.2


center_x = width /2
center_y = height / 2

K = np.array([[focal_length, 0, center_x],
              [0, focal_length, center_y],
              [0, 0, 1]])


frame1 = read_specific_frame("test",0)
frame2 = read_specific_frame("test",1)

kp1, des1 = extract_sift_features(frame1)
kp2, des2 = extract_sift_features(frame2)

matches = match_features(des1, des2)

points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# Calculate Essential Matrix
F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
E, mask = cv2.findEssentialMat(points1, points2, K)



print("Fundamental Matrix:\n", F)

pp = (center_x, center_y)  # your camera's principal point

# Recover the pose from the essential matrix
_, R, t, mask = cv2.recoverPose(E, points1, points2, focal=focal_length, pp=pp)

print("Rotation Matrix:\n", R)
print("Translation Vector:\n", t)

U, S, Vt = np.linalg.svd(F)

# The epipoles are the last column of Vt and the last column of U
e1 = Vt[-1, :]
e2 = U[:, -1]

# Normalize the epipoles (since they are in homogeneous coordinates)
e1 = e1 / e1[2]
e2 = e2 / e2[2]

print(f"Epipole in first image: {[format(coord, '.4f') for coord in e1]}")
print(f"Epipole in second image: {[format(coord, '.4f') for coord in e2]}")


epipole_radius = 20
epipole_color = (0, 0, 255)  # Red color in BGR
center_color = (255, 0, 0)  # Red color in BGR
epipole_thickness = 10

# Draw the epipole on the first frame
center = (int(center_x),int(center_y))
epipole1 = (int(e1[0]), int(e1[1]))
cv2.circle(frame1, epipole1, epipole_radius, epipole_color, epipole_thickness)
cv2.circle(frame1, center, epipole_radius, center_color, epipole_thickness)

# Draw the epipole on the second frame
epipole2 = (int(e2[0]), int(e2[1]))
cv2.circle(frame2, epipole2, epipole_radius, epipole_color, epipole_thickness)
cv2.circle(frame2, center, epipole_radius, center_color, epipole_thickness)




#cv2.imshow("Frame 1 with Epipole (Scaled)", scaled_frame1)
#cv2.imshow("Frame 2 with Epipole (Scaled)", scaled_frame2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

img_matches = draw_matches(frame1, kp1, frame2, kp2, matches, mask)
scaled_img_matches = cv2.resize(img_matches, (img_matches.shape[1] // 4, img_matches.shape[0] // 4))
cv2.imshow("Matched Features", scaled_img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Release the video capture object and close the windows
#cv2.destroyAllWindows()


