"""
Filename:  Readme.txt
Date:      03/28/2020
Author:    Rutvij Shah
Email:     rutvij.shah@utdallas.edu
Course:    CS6384.001 (Computer Vision)
Version:   1.0
Copyright: 2022, All Rights Reserved
Description:
    Lane Line Detection Project (Python 3.10)
"""

The algorithms run instantaneously, but to further speed up processing and separate processing
from image display, Python's multiprocessing facilities are used.

All files within a folder are opened by default and any dotfiles are ignored.

Requirements for the project are:
- Python 3.10
- Open CV
- Matplotlib
- NumPy
- Pillow
- Sci-Kit Learn

If a give requirement isn't found, import guards will report the missing library and provide
the necessary command to install it.



Detection Approaches:


1] Lane line detection
    - Resize image to 400 x 300 px
    - Dilate the image with a cross-kernel of size 3x3
        - fill in noisy pixels with surrounding majority
    - Increase the contrast of the image by a factor of 2
        - saturate the colors to define edges better
    - Convert the image to grayscale
        - pre-processing for canny edge detector which works on single channel images
    - Apply a bilateral filter to the image
        - reduce noise in the image by applying a bilateral image to blur it
    - Apply canny edge detector to find edges
        - canny parameters are calculated using Otsu's method of thresholding.
    - Find contours in the canny edge detected image
        - any contours having an area smaller than 10 px^2 are removed
   - Apply probabilistic Hough Transform to obtain straight lines within the image
   - Cluster the lines found using probabilistic Hough transform to remove potentially similar
   lines (DBSCAN is used for clustering)
   - Draw the longest line from each cluster and display the final image.

2] Stop Sign detection
    - Create a red mask in the HSV range of red (lower and upper ranges)
    - Apply the red mask over the image to isolate red only areas
    - Apply Gaussian blur to the image using a 5x5 kernel
    - Convert the BGR colorspace image to grayscale
    - Apply thresholding to the gray scale image to convert it to a binary grayscale image
    - find contours in the binary image
    - calculate the minimum radius a stop sign could be (heuristic)
        - min radius is either 25px or 5% of the smaller image dimension, whichever is greater.
    - for each contour find the minimum enclosing circle, and isolate the largest such circle
        - in the process of isolation, ensure that the circular area selected has a red pixel
        percentage of more than 50%
    - if a stop sign is detected, return image with it circled else return None
