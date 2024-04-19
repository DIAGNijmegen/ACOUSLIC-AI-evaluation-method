import cv2
import numpy as np


def compute_intersection_with_fov(ellipse_mask, fov_mask):
    # Convert fov_mask to binary where non-zero values are set to 255
    fov_mask_binary = np.where(fov_mask != 0, 255, 0).astype(np.uint8)

    # Find contours of the fov_mask
    contours_fov, _ = cv2.findContours(
        fov_mask_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create an image to draw the contours of fov_mask
    contour_image_fov = np.zeros_like(fov_mask)
    cv2.drawContours(contour_image_fov, contours_fov, -
                     1, 255, 1)  # Fill the contour

    # Find the intersection between ellipse_mask and contour_image_fov
    intersection_mask = ellipse_mask * (contour_image_fov != 0)

    # Find contours of the intersection_mask
    contours_intersection, _ = cv2.findContours(
        intersection_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return intersection_mask, contours_intersection


def find_closest_contour_point(point, contour):
    # This function finds the closest point on the contour to the given point.
    # The point and contour points are expected to be 2D (x, y).
    min_dist = np.inf
    closest_point = None
    for cnt_point in contour:
        # Ensure cnt_point is a single point in 2D
        cnt_point_2d = cnt_point[0] if len(cnt_point.shape) == 2 else cnt_point

        dist = np.linalg.norm(np.array(cnt_point_2d) - np.array(point))
        if dist < min_dist:
            min_dist = dist
            closest_point = cnt_point_2d

    return np.array(closest_point)  # Return as a 2D numpy array


def find_nearest_contour_index(contour, point):
    # Find the index of the nearest point in contour to the given point
    distances = np.linalg.norm(contour[:, 0, :] - point, axis=1)
    nearest_index = np.argmin(distances)
    return nearest_index


def find_contour_segment(contour, idx1, idx2):
    # Extract two possible segments between idx1 and idx2
    segment_direct = contour[idx1:idx2 +
                             1] if idx1 <= idx2 else contour[idx2:idx1+1]
    segment_wrap = np.concatenate(
        (contour[idx2:], contour[:idx1+1])) if idx1 <= idx2 else np.concatenate((contour[idx1:], contour[:idx2+1]))

    # Determine which segment to return
    # For simplicity, we choose the shorter segment; adjust logic here if needed for your specific case
    return segment_direct if cv2.arcLength(segment_direct, False) < cv2.arcLength(segment_wrap, False) else segment_wrap


def find_arc_midpoint(segment):
    total_length = cv2.arcLength(segment, False)
    half_length = total_length / 2

    accumulated_length = 0
    for i in range(1, len(segment)):
        p1 = segment[i - 1][0]
        p2 = segment[i][0]
        segment_length = np.linalg.norm(p1 - p2)

        if accumulated_length + segment_length >= half_length:
            # Find the point along the current segment that divides the total length in half
            excess_length = accumulated_length + segment_length - half_length
            ratio = (segment_length - excess_length) / segment_length
            midpoint = p1 + ratio * (p2 - p1)
            return midpoint

        accumulated_length += segment_length

# Function to find the equation of a line given two points


def line_equation(p1, p2):
    m = (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[0] - \
        p1[0] != 0 else float('inf')
    c = p1[1] - m * p1[0] if m != float('inf') else p1[0]
    return m, c

# Function to get the perpendicular intersection point with the contour


def perpendicular_intersection(m, c, point, contour):
    if m == 0:  # Horizontal line case
        perp_line_m = float('inf')
    elif m == float('inf'):  # Vertical line case
        perp_line_m = 0
    else:
        perp_line_m = -1 / m

    perp_line_c = point[1] - perp_line_m * \
        point[0] if perp_line_m != float('inf') else point[0]

    # Find the closest contour point - approximate method
    min_distance = float('inf')
    closest_point = None
    for cnt_point in contour:
        # Ensure cnt_point is accessed correctly
        x, y = cnt_point[0] if len(cnt_point.shape) > 1 else cnt_point

        if perp_line_m == float('inf'):  # Perpendicular line is vertical
            distance = abs(x - perp_line_c)
        else:
            # Distance from point to line
            distance = abs(perp_line_m * x - y + perp_line_c) / \
                np.sqrt(perp_line_m**2 + 1)

        if distance < min_distance:
            min_distance = distance
            closest_point = (x, y)  # Ensure this is a tuple or 2D point

    return np.array(closest_point)


def get_extremes_truncated_boundary(binary_mask_fov, mask_fov):
    intersection, contours_intersection = compute_intersection_with_fov(
        binary_mask_fov, mask_fov)
    contours_intersection = np.squeeze(contours_intersection)
    contours_intersection = np.squeeze(contours_intersection)
    # Find topmost and bottommost points - these could serve as start and end points for vertical truncation
    # Assuming contours_intersection is a numpy array of shape (n, 2)
    topmost_index = contours_intersection[:, 1].argmin()
    bottommost_index = contours_intersection[:, 1].argmax()

    topmost_point = tuple(contours_intersection[topmost_index])
    bottommost_point = tuple(contours_intersection[bottommost_index])
    # Calculate the midpoint
    middle_point = ((topmost_point[0] + bottommost_point[0]) / 2,
                    (topmost_point[1] + bottommost_point[1]) / 2)
    return topmost_point, bottommost_point, middle_point


def get_contour_points(binary_mask):
    # Find the contours in the binary mask
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        print("No contours found in the binary mask.")
        return []
    # Assuming the largest contour corresponds to the bubble
    largest_contour = max(contours, key=cv2.contourArea)
    # Reshape the contour array
    return largest_contour  # .reshape(-1, 2)


def fit_ellipse_to_points(points, shape, thickness=-1):
    # Fit an ellipse to the points using OpenCV and return the parameters and the fitted ellipse
    ellipse = cv2.fitEllipse(points)
    # Drawing the ellipse
    fitted_ellipse = np.zeros(shape, dtype=np.uint8)
    # The ellipse variable contains: (center (x, y), (major axis length, minor axis length), angle of rotation)
    cv2.ellipse(fitted_ellipse, ellipse, 255, thickness)
    return ellipse, fitted_ellipse


def get_non_truncated_ellipse_contours(ellipse_mask, fov_mask):
    # Convert fov_mask to binary where non-zero values are set to 255
    fov_mask_binary = np.where(fov_mask != 0, 255, 0).astype(np.uint8)
    ellipse_mask = np.where(ellipse_mask != 0, 255, 0).astype(np.uint8)

    # Find contours of the fov_mask and binary_mask
    contours_fov, _ = cv2.findContours(
        fov_mask_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_binary, _ = cv2.findContours(
        ellipse_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create an image to draw the contours of fov_mask
    contour_image_fov = np.zeros_like(fov_mask)
    contour_mask = np.zeros_like(fov_mask)
    cv2.drawContours(contour_image_fov, contours_fov, -
                     1, 255, 1)  # Fill the contour
    cv2.drawContours(contour_mask, contours_binary, -
                     1, 255, 1)  # Fill the contour

    # Find the intersection between ellipse_mask and contour_image_fov
    non_truncated_contour_mask = np.logical_and(
        contour_mask, np.logical_not(contour_image_fov))

    # Find contours of the intersection_mask
    non_truncated_contour_mask = np.where(
        non_truncated_contour_mask != 0, 255, 0).astype(np.uint8)
    contours_non_truncated = get_contour_points(non_truncated_contour_mask)
    return non_truncated_contour_mask, contours_non_truncated



