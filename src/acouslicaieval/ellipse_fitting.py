import math

import cv2
import numpy as np
import SimpleITK as sitk

MASK_FOV = sitk.GetArrayFromImage(sitk.ReadImage("data/fov_mask.mha"))


def fit_ellipses(binary_mask, thickness=1):
    binary_mask = binary_mask.copy()
    # Ensure binary mask is of type uint8
    binary_mask = np.where(binary_mask != 0, 255, 0).astype(np.uint8)
    mask_fov = np.where(MASK_FOV != 0, 255, 0).astype(np.uint8)

    # Pad the mask to allow for interpolation of ellipses that would extend beyond the image boundaries
    # (e.g., half-ellipses at the bottom of fov)
    binary_mask_padded = zero_pad_image(binary_mask, pad_width=200)
    mask_fov_padded = zero_pad_image(mask_fov, pad_width=200)

    # Apply FOV masking
    binary_mask_fov = (binary_mask_padded * mask_fov_padded).astype(np.uint8)

    # Get non-truncated contours (those not overlapping with fov mask contours)
    _, contours_non_truncated = get_non_truncated_ellipse_contours(
        binary_mask_fov, mask_fov_padded)

    # The non-truncated contours are the ones that define the ellipse
    if len(contours_non_truncated) > 5:   # At least 5 points are required to fit an ellipse
        ellipse = cv2.fitEllipse(contours_non_truncated)
        if not any(math.isnan(param) for param in ellipse[1]):
            a = ellipse[1][0] / 2 # Semi-major axis
            b = ellipse[1][1] / 2 # Semi-minor axis

            # Calculate the estimated circumference of the ellipse
            circumference = ellipse_circumference(a, b)

            # Draw the fitted ellipse on a separate image and store it
            fitted_ellipse_mask = np.zeros_like(binary_mask_fov)
            cv2.ellipse(fitted_ellipse_mask, ellipse, 1, thickness)
            # Remove padding
            fitted_ellipse_mask = fitted_ellipse_mask[200:-200, 200:-200]

            return ellipse, circumference, fitted_ellipse_mask
    print("No ellipse found")
    return None, None, None


def create_ellipse(x_center, y_center, a, b, angle, size_x, size_y):
    y, x = np.ogrid[-y_center:size_x - y_center, -x_center:size_y - x_center]
    ellipse_mask = ((x * np.cos(angle) + y * np.sin(angle)) / a) ** 2 \
        + ((y * np.cos(angle) - x * np.sin(angle)) / b) ** 2 <= 1
    return ellipse_mask.astype(np.uint8) * 255


def ellipse_circumference(a, b):
    """
    Approximate the circumference of an ellipse using Ramanujan's first formula.

    Parameters:
    a (float): Semi-major axis of the ellipse.
    b (float): Semi-minor axis of the ellipse.

    Returns:
    float: The approximate circumference of the ellipse.
    """
    return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))


def pixels_to_mm(pixels, conversion_factor):
    """
    Converts a value in pixels to millimeters using the provided conversion factor.

    Args:
        pixels (float): The value in pixels to be converted.
        conversion_factor (float): The conversion factor from pixels to millimeters.

    Returns:
        float: The converted value in millimeters.
    """
    return pixels * conversion_factor


def zero_pad_image(image, pad_width):
    """
    Add zero padding to an image or a 2D mask. Adjusts padding based on the input dimensionality.

    :param image: Input image or mask as a NumPy array.
    :param pad_width: The width of the padding around the image or mask. 
                      This could be a single integer or a tuple (pad_top, pad_bottom, pad_left, pad_right).
    :return: Zero-padded image or mask.
    """
    if isinstance(pad_width, int):
        # Apply the same padding to all sides
        pad_width = image.ndim * [pad_width]

    # Use np.pad to add zeros around the image or mask
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)

    return padded_image


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
    return largest_contour


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
