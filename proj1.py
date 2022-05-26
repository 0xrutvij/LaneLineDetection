"""
Filename:  proj1.py
Date:      03/28/2020
Author:    Rutvij Shah
Email:     rutvij.shah@utdallas.edu
Course:    CS6384.001 (Computer Vision)
Version:   1.0
Copyright: 2022, All Rights Reserved
Description:
    Lane Line Detection Project (Python 3.10)
"""
from __future__ import annotations

import argparse
import multiprocessing
import random as rng
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Callable, Iterable, TypeVar, Union
from uuid import uuid4


def print_import_failure(package, distribution_name):
    print(f"{package} not installed, please use pip install {distribution_name} to install it.")
    exit(-1)


try:
    # noinspection PyUnresolvedReferences
    import cv2 as cv
except ImportError:
    print_import_failure("Open CV", "opencv-python")

try:
    # noinspection PyUnresolvedReferences
    import matplotlib.pyplot as plt
except ImportError:
    print_import_failure("Matplotlib", "matplotlib")

try:
    # noinspection PyUnresolvedReferences
    import numpy as np
except ImportError:
    print_import_failure("NumPy", "numpy")

try:
    # noinspection PyUnresolvedReferences
    from PIL import ImageEnhance, Image
except ImportError:
    print_import_failure("Pillow", "Pillow")

try:
    # noinspection PyUnresolvedReferences
    from sklearn.cluster import DBSCAN
    # noinspection PyUnresolvedReferences
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print_import_failure("Sci-Kit Learn", "scikit-learn")


Point = Tuple[int, int]
HomogeneousList = Union[np.ndarray, List]
T, S = map(TypeVar, ['T', 'S'])

LOWER_WHITE = np.array([0, 0, 200])
UPPER_WHITE = np.array([180, 60, 255])

LOWER_RED1 = np.array([0, 70, 50])
UPPER_RED1 = np.array([10, 255, 255])

LOWER_RED2 = np.array([170, 70, 50])
UPPER_RED2 = np.array([180, 255, 255])

LOWER_YELLOW = np.array([11, 60, 0])
UPPER_YELLOW = np.array([40, 255, 255])

LOWER_GREEN = np.array([35, 50, 70])
UPPER_GREEN = np.array([89, 255, 255])


def show_image(image: np.ndarray) -> None:
    """Display an image using Matplotlib"""
    plt.figure()
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.draw()


def bgr_to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert a BGR space image to HSV"""
    return cv.cvtColor(image, cv.COLOR_BGR2HSV)


def bgr_to_gray(image: np.ndarray) -> np.ndarray:
    """Convert a BGR space image to Grayscale"""
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def get_mask(image: np.ndarray, lower_thresh: np.ndarray, upper_thresh: np.ndarray) -> np.ndarray:
    """Get a mask for the given image, using the upper and lower bound HSV values provided"""
    hsv = bgr_to_hsv(image)
    return cv.inRange(hsv, lower_thresh, upper_thresh)


def combine_masks(*masks) -> np.ndarray:
    """Sum give masks to form an aggregate mask"""
    return sum(masks)


def mask_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply the given mask to an image and return the result"""
    return cv.bitwise_and(image, image, mask=mask)


def resize_and_crop(image: np.ndarray, square_px: int, vertical_crop_ratio: float) -> np.ndarray:
    """Resize an image to a px * px square, and crop vertical_crop_ratio worth of the image's height"""
    size = (square_px, square_px)
    cropped_image = cv.resize(image, size)
    vertical_crop_px = int(vertical_crop_ratio * square_px)
    return cropped_image[vertical_crop_px:, :, :]


def increase_contrast(image: np.ndarray, scaling_factor: float) -> np.ndarray:
    """Increase the contrast of an image by given scaling factor"""
    pil_image = Image.fromarray(image)
    contrast_enhanced_pil_image = ImageEnhance.Contrast(pil_image).enhance(scaling_factor)
    # noinspection PyTypeChecker
    return np.asarray(contrast_enhanced_pil_image)


def _apply_canny_edge_detector(
        blurred_gray_image: np.ndarray,
        thresh_ratio: float = 0.5,
        use_l2_grad: bool = True
) -> np.ndarray:
    """Apply canny edge detector to given grayscale image, threshold ratio is the ratio of upper and lower
    threshold calculated using Otsu's Method."""
    upper_threshold, _ = cv.threshold(blurred_gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    lower_threshold = upper_threshold * thresh_ratio
    return cv.Canny(blurred_gray_image, lower_threshold, upper_threshold, L2gradient=use_l2_grad)


def apply_canny_edge_detector(
        color_image: np.ndarray,
        bilateral_filter_args: List = None,
        min_contour_area: float = 10,
        canny_thresh_ratio: float = 0.5,
        use_l2_grad: bool = True
) -> np.ndarray:
    """Apply canny edge detector to a color image, if bilateral filter args aren't provided they will default to
    [8, 200, 200]. The filtered image is then passed to the edge detector. The contours thus formed are then filtered
    so that those having area less than min_contour_area are removed to reduce noise."""
    gray_image = bgr_to_gray(color_image)

    # d: pixel neighborhood, sigma color and sigma space
    filter_args = [8, 200, 200]
    if bilateral_filter_args is not None:
        len_condition = len(bilateral_filter_args) == 3
        type_condition = all(map(lambda x: isinstance(x, int), bilateral_filter_args))
        if len_condition and type_condition:
            filter_args = bilateral_filter_args

    blurred_image = cv.bilateralFilter(gray_image, *filter_args)

    canny = _apply_canny_edge_detector(blurred_image, canny_thresh_ratio, use_l2_grad)

    contours = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for contour in contours:
        area = cv.contourArea(contour)
        if area < min_contour_area:
            cv.drawContours(canny, [contour], 0, (0, 0, 0), -1)

    return canny


def process_probabilistic_hough_lines(
        lines: np.ndarray,
        slope_lower: float, slope_upper: float) -> Tuple[List, List]:
    """Find the probabilistic Hough lines for the given edge detected image and filter out lines having
    slopes lower than slope_lower as well as lines with slope greater than slope_upper"""
    lines = lines.squeeze()

    points = []
    slopes_and_intercepts = []

    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line
        dy = y2 - y1
        diff = x2 - x1
        dx = diff if diff != 0 else pow(10, -10)
        slope = dy/dx
        abs_slope = abs(slope)
        intercept = y1 - slope * x1
        # line is too flat or if a line is almost vertical
        if abs_slope < slope_lower or abs_slope > slope_upper:
            continue
        else:
            points.append([(x1, y1), (x2, y2)])
            slopes_and_intercepts.append([slope, round(intercept)])

    return points, slopes_and_intercepts


def _cluster_lines(clustering_input: HomogeneousList, params: Dict = None) -> HomogeneousList:
    """Cluster given lines using DBSCAN algorithm and given parameters"""
    X = StandardScaler().fit_transform(clustering_input)
    clustering = DBSCAN(**params).fit(X)
    return clustering.labels_


def cluster_lines(
        points: HomogeneousList,
        slopes_intercepts: HomogeneousList,
        clustering_algo_params: Dict = None
) -> Dict:
    """Clusters lines on the basis of their slopes and intercepts, returns the clusters as a dictionary. If no
    parameters are provided, an eps = 0.09 and min_samples of 2 are used for DBSCAN."""

    params = {"eps": 0.09, "min_samples": 2}
    if clustering_algo_params is not None:
        params = clustering_algo_params

    labels = _cluster_lines(slopes_intercepts, params)
    clusters = defaultdict(lambda: {"points": list(), "slopes": list()})

    for line, slope_interp, label_num in zip(points, slopes_intercepts, labels):
        if label_num == -1:
            label_num = uuid4()

        clusters[label_num]["points"].append(line)
        clusters[label_num]["slopes"].append(slope_interp)

    return clusters


def draw_clustered_lines(
        image: np.ndarray,
        clusters: Dict,
        debug: bool = False,
        thickness: int = 10,
        slope_skip_criterion: float = 1/3,
        image_size: int = 400
) -> Tuple[np.ndarray, int]:
    """Given a dictionary of line clusters, find the longest line in each and draw it on the given image. If two
    clusters have longest lines with slope within skip_criterion of each other one of the lines is skipped."""
    slopes_seen = []
    lines_skipped = 0
    lines_drawn = 0
    for cluster in clusters.values():
        color = [rng.randint(0, 255) for _ in range(3)]
        skip_cluster = False
        points = cluster["points"]
        slope_interps = cluster["slopes"]

        np_points = np.array(points)
        index_longest_line = np.argmax(list(map(np.linalg.norm, np_points)))
        slope, intercept = slope_interps[index_longest_line]

        for slp in slopes_seen:
            diff = abs(slope - slp)
            if diff < slope_skip_criterion:
                skip_cluster = True

        if skip_cluster:
            lines_skipped += 1
            continue
        else:
            slopes_seen.append(round(slope, 3))

        max_bound = image_size + 100
        start = 0, int(intercept)
        end = max_bound, int((slope * max_bound) + intercept)
        cv.line(image, start, end, color, thickness, cv.LINE_AA)
        lines_drawn += 1

    if debug:
        print(f"Slopes{sorted(slopes_seen)}")
        print(f"Lines Printed {lines_drawn}, Skipped {lines_skipped}")

    return image, lines_drawn


def detect_lane(image: np.ndarray) -> Tuple[int, np.ndarray]:
    """Given an image, detect all lane lines within it and return the number of lines detected
    as well as the image with lines drawn over it."""
    image_size = 400
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

    cropped_image = resize_and_crop(image, image_size, 0.25)

    dilated_image = cv.dilate(cropped_image, kernel, iterations=1)

    contrasted_image = increase_contrast(dilated_image, 2.0)

    canny_image = apply_canny_edge_detector(
        contrasted_image,
        bilateral_filter_args=[8, 200, 200],
        min_contour_area=10,
        canny_thresh_ratio=0.5,
        use_l2_grad=True
    )

    probabilistic_lines = cv.HoughLinesP(
        canny_image,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=100,
        maxLineGap=30
    )

    points, slope_intercepts = process_probabilistic_hough_lines(probabilistic_lines, 0.25, 20)
    line_clusters = cluster_lines(points, slope_intercepts, {"eps": 0.09, "min_samples": 2})
    final_image, lines_detected = draw_clustered_lines(cropped_image, line_clusters,
                                                       thickness=10, slope_skip_criterion=1/3, image_size=image_size)

    return lines_detected, final_image


def detect_stop_sign(image: np.ndarray) -> np.ndarray | None:
    """Given an image, detect whether or not it has a stop sign within it, if it does, return the image
    with the stop sign circled in blue."""
    red_mask1 = get_mask(image.copy(), LOWER_RED1, UPPER_RED1)
    red_mask2 = get_mask(image.copy(), LOWER_RED2, UPPER_RED2)
    ovr_mask = red_mask2 + red_mask1
    red_only_image = mask_image(image, ovr_mask)

    blurred_image = cv.GaussianBlur(red_only_image, (5, 5), 0)

    gray_image = bgr_to_gray(blurred_image)

    binary_image = cv.threshold(gray_image, 20, 255, cv.THRESH_BINARY)[1]
    # show_image(binary_image)

    contours, hierarchy = cv.findContours(binary_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    max_circ_image = image.copy()

    min_allowed_radius = max(min(image.shape[:2]) * 0.05, 25)

    current_max_radius = min_allowed_radius

    for contour in contours:
        (x, y), radius = cv.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        if radius > current_max_radius:
            image_indices = np.indices(binary_image.shape)
            cent = np.array([center[1], center[0]])
            diffs = image_indices - cent[:, None, None]
            dists = np.linalg.norm(diffs, axis=0)
            circle_region = binary_image[np.where(dists <= radius)]

            total_pixels = circle_region.size
            white_pixels = np.sum(circle_region > 200)

            if (white_pixels + 1)/(total_pixels + 1) > 0.5:
                max_circ_image = image.copy()
                max_circ_image = cv.circle(max_circ_image, center, radius, (255, 0, 0), 4)
                current_max_radius = radius

    if not np.array_equal(max_circ_image, image):
        return max_circ_image
    else:
        return None


###########################################################################


def runon_image(path):
    frame = cv.imread(path)
    num_lanes, final_image = detect_lane(frame)
    return num_lanes, final_image, path


def runon_image_stop(path):
    image = cv.imread(path)
    return detect_stop_sign(image), path


def create_mapper(mapping_func: Callable[[Callable, Iterable], Iterable]
                  ) -> Callable[[Callable, Iterable], Iterable]:

    def mapper(func: Callable[[T], S], f_args: Iterable[T]) -> Iterable:
        return mapping_func(func, f_args)

    return mapper


def runon_folder(path: str, run_parallel: bool = False) -> int:
    files = None
    img_folder = Path(path)

    if img_folder.is_dir():
        files = [str(path) for path in Path(img_folder).glob("*") if path.is_file() and not path.name.startswith(".")]

    total_detections = 0

    if run_parallel:
        process_pool = multiprocessing.Pool(multiprocessing.cpu_count())
        mapper = create_mapper(process_pool.imap_unordered)
    else:
        process_pool = None
        mapper = create_mapper(map)

    detected_images = []
    for f_detections, f_image, f in mapper(runon_image, files):
        print(f"Processed image {f}")
        if f_image is not None:
            detected_images.append((f_image, f))
        total_detections += f_detections

    stop_detected_images = []
    for f_image, f in mapper(runon_image_stop, files):
        if f_image is not None:
            print(f"Found stop sign in {f}")
            stop_detected_images.append((f_image, f))

    for img, name in detected_images + stop_detected_images:
        cv.imshow(name, img)
        plt.imsave(f"output/{Path(name).stem}.png", cv.cvtColor(img, cv.COLOR_BGR2RGB))
        cv.waitKey(0)
        cv.destroyAllWindows()

    cv.destroyAllWindows()
    if process_pool is not None:
        process_pool.close()

    return total_detections


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-folder', help="requires path")
    args = parser.parse_args()
    folder = args.folder
    if folder is None:
        print("Folder path must be given \n Example: python proj1.py -folder images")
        sys.exit()

    if folder is not None:
        all_detections = runon_folder(folder)
        print("total of ", all_detections, " detections")
