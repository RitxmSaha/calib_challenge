import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

focal_length = 910
width = 1164
height = 874
center_x = width/2
center_y = height/2
pp = (center_x, center_y)


K = np.array([[focal_length, 0, center_x],
              [0, focal_length, center_y],
              [0, 0, 1]])
K_inv = np.linalg.inv(K)

def read_specific_frame(video_directory, frame_number):
    frame_path = f"./{video_directory}/{frame_number}.jpg"

    # Check if the frame exists
    if not os.path.exists(frame_path):
        print(f"Frame {frame_number} does not exist in the directory.")
        return None

    frame = cv2.imread(frame_path)

    return frame

def create_black_pixel_mask(image, threshold=0):
    black_pixels = (image == 0)
    kernel = np.ones((threshold*2+1, threshold*2+1), np.uint8)
    mask = cv2.dilate(black_pixels.astype(np.uint8), kernel, iterations=1)
    return mask

def is_near_black_pixel(point, mask):
    x, y = int(point[0]), int(point[1])
    return mask[y, x] == 1

def calculateEpipole(F):
    U, S, Vt = cv2.SVDecomp(F)
    epipole = Vt.T[:, -1]

    epipole = epipole / epipole[-1]

    epipole_x, epipole_y = int(epipole[0]), int(epipole[1])
  
    return epipole_x, epipole_y

sift = cv2.SIFT_create(contrastThreshold = 0.01, edgeThreshold=250, sigma=1.6)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('epipoles.mp4', fourcc, 10.0, (width, height))

dir = "0"
data = pd.read_csv('../labeled/'+dir+'.txt', sep=' ', header=None, names=['pitch', 'yaw'])

start_frame = 0
end_frame = 200

first_frame = read_specific_frame(dir, start_frame)
first_gray  = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
keypoints, descriptors = sift.detectAndCompute(first_gray, None)

second_matches = (keypoints, descriptors)
for frame in range(start_frame,end_frame):
    real_yaw = data['yaw'][frame]
    real_pitch = data['pitch'][frame]
    real_epipole_x = np.tan(real_yaw) * focal_length + center_x
    real_epipole_y = center_y - np.tan(real_pitch) * focal_length
    if(np.isnan(real_epipole_x)):
        continue
    frame1 = read_specific_frame(dir, frame)
    frame2 = read_specific_frame(dir, frame+1)

    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    mask1 = create_black_pixel_mask(gray1)
    mask2 = create_black_pixel_mask(gray2)

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = second_matches
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    second_matches = (keypoints2, descriptors2)
    print("Frame Number: " + str(frame))

    matches = bf.knnMatch(descriptors1,descriptors2,k=2)
    print("num before lowe: "+str(len(matches)))
    good_matches = []
    ratio_thresh = 0.75
    for m,n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append([m])
    matches = good_matches

    print("num after lowe: "+str(len(matches)))

    filtered_matches = []
    for match in matches:
        m = match[0]
        pt1 = keypoints1[m.queryIdx].pt
        pt2 = keypoints2[m.trainIdx].pt

        (x1, y1) = pt1
        (x2, y2) = pt2
        if not (is_near_black_pixel(pt1, mask1) or is_near_black_pixel(pt2, mask2)):
            filtered_matches.append(m)
    matches = filtered_matches

    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    K_inv = np.linalg.inv(K)

    E, inliers = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, prob=0.9999, threshold=1)
    F_corr = np.linalg.inv(K).T @ E @ np.linalg.inv(K)

    calc2_x, calc2_y = calculateEpipole(F_corr)

    print("Epipole in the first image: essential: ", calc2_x, calc2_y)

    # Filter matches based on inliers
    good_matches = [m for m, inlier in zip(matches, inliers.ravel()) if inlier]
    for m in good_matches:
        img1_idx = m.queryIdx
        img2_idx = m.trainIdx

        # x - columns, y - rows
        (x1, y1) = keypoints1[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt

        extension_factor = 100
        direction = -1 *np.array([x2 - x1, y2 - y1])
        extended_point = np.array([x1, y1]) + extension_factor * direction


        #draw feature and their trajectory on frame
        cv2.line(frame1, (int(x1), int(y1)), (int(extended_point[0]), int(extended_point[1])), (255, 0, 0), 1)
        cv2.circle(frame1, (int(x1), int(y1)), 5, (0, 255, 0), -1)
    
    #draw real and calculated epipoles on frame
    cv2.circle(frame1, (int(real_epipole_x), int(real_epipole_y)), 40, (255, 255, 255), -1) 
    cv2.circle(frame1, (int(calc2_x), int(calc2_y)), 10, (255, 255, 0), -1)
    video_writer.write(frame1)
video_writer.release()