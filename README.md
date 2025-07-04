# SIFT-Based-Feature-Detection-and-Matching-in-Rotated-Images
Implemented SIFT-based keypoint detection and matching between original and 45° rotated image pairs. Applied Lowe’s ratio test and visualized the top 5 matches to demonstrate rotation-invariant feature matching in image pairs using OpenCV.

# Feature Detection and Matching using SIFT

## Overview
This project implements image feature extraction and matching using the SIFT algorithm to evaluate the consistency of keypoints between original and rotated image pairs.

## Features
- Loads two original images from the "Dataset/" folder
- Rotates each image by 45 degrees and saves them
- Extracts SIFT keypoints and descriptors from all four images
- Matches original images to their rotated versions using BFMatcher
- Applies Lowe’s ratio test to refine matches
- Sorts and visualizes the top 5 matches per image pair
- Saves match visualizations to the dataset folder

## How It Works
1. **Image Preparation:** Reads `original_1.jpg` and `original_2.jpg`, rotates each by 45°, saves as `rotated_1.jpg` and `rotated_2.jpg`.
2. **Feature Detection:** Uses SIFT to detect keypoints and descriptors.
3. **Feature Matching:** Matches original images with their rotated versions using k-NN and Lowe's ratio test.
4. **Visualization:** Top 5 matches are drawn and saved as `image1_match.jpg` and `image2_match.jpg`.

## Requirements
- Python 3
- OpenCV (`cv2`) with SIFT support
- NumPy
- Matplotlib
- PIL

## Run the Code
Place your `original_1.jpg` and `original_2.jpg` in the `Dataset/` folder, then run:

```bash
python main.py
