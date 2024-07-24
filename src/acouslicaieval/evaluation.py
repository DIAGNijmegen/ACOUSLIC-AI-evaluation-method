
import numpy as np
import SimpleITK
from evalutils import ClassificationEvaluation
from evalutils.io import SimpleITKLoader
from evalutils.validators import (
    UniqueImagesValidator,
    UniquePathIndicesValidator,
)

from acouslicaieval.ellipse_fitting import MASK_FOV, fit_ellipses, pixels_to_mm

# Maximum frame tolerance for the selected frame with respect to the next optimal
# frame in the ground truth, within the same sweep
MAX_FRAME_TOLERANCE = 15


class FetalAbdomenSegmentationEval(ClassificationEvaluation):
    """Evaluation class for fetal abdomen segmentation masks."""

    def __init__(self):
        super().__init__(
            file_loader=SimpleITKLoader(),
            validators=(
                # NumberOfCasesValidator(num_cases=2),
                UniquePathIndicesValidator(),
                UniqueImagesValidator(),
            ),
        )

    def score_case(self, *, gt_array, pred, frame):
        """
        Compute overlap metrics for a single case.

        Args:
        gt_array (ndarray): Ground truth segmentation array.
        pred (Image): Predicted segmentation image.
        frame (int): Frame number to evaluate.

        Returns:
        dict: A dictionary containing various overlap metrics.
        """
        # Preprocess ground truth array to ensure binary segmentation
        gt = gt_array[frame].copy()
        gt[gt > 0] = 1
        gt = SimpleITK.GetImageFromArray(gt)
        pred = SimpleITK.GetImageFromArray(pred.copy())

        # Cast to the same type for evaluation
        caster = SimpleITK.CastImageFilter()
        caster.SetOutputPixelType(SimpleITK.sitkUInt8)
        caster.SetNumberOfThreads(1)
        gt = caster.Execute(gt)
        pred = caster.Execute(pred)

        # Apply FOV masking
        mask_fov_itk = SimpleITK.GetImageFromArray(MASK_FOV)
        mask_fov_itk = caster.Execute(mask_fov_itk)
        gt = SimpleITK.Mask(gt, mask_fov_itk)
        pred = SimpleITK.Mask(pred, mask_fov_itk)

        # Initialize overlap measures and compute metrics
        overlap_measures = SimpleITK.LabelOverlapMeasuresImageFilter()
        overlap_measures.SetNumberOfThreads(1)
        overlap_measures.Execute(gt, pred)
        metrics = {
            'FalseNegativeError': overlap_measures.GetFalseNegativeError(),
            'FalsePositiveError': overlap_measures.GetFalsePositiveError(),
            'MeanOverlap': overlap_measures.GetMeanOverlap(),
            'UnionOverlap': overlap_measures.GetUnionOverlap(),
            'VolumeSimilarity': overlap_measures.GetVolumeSimilarity(),
            'JaccardCoefficient': overlap_measures.GetJaccardCoefficient(),
            'DiceCoefficient': overlap_measures.GetDiceCoefficient(),
            'DiceCoefficientNearestFrame': overlap_measures.GetDiceCoefficient(),
            'NearestAnnotatedFrame': frame,
            'DiceCoefficientSoft': overlap_measures.GetDiceCoefficient(),
        }

        # Instantiate Hausdorff distance calculator
        hausdorff_calculator = SimpleITK.HausdorffDistanceImageFilter()
        hausdorff_calculator.SetNumberOfThreads(1)

        # Compute Hausdorff distance on selected ground truth frame if it contains an annotation
        if np.any(gt) and np.any(pred):
            # Compute the distance to the next optimal frame
            hausdorff_calculator.Execute(gt, pred)
            hausdorff_distance = hausdorff_calculator.GetHausdorffDistance()
        elif not np.any(SimpleITK.GetArrayFromImage(pred)):
            # If the prediction is empty, set the Hausdorff distance to maximum sweep width (744)
            # scaled by the frame tolerance
            hausdorff_distance = 744 * MAX_FRAME_TOLERANCE
            # print('Prediction is empty. Setting Hausdorff distance to maximum sweep width (744) scaled by the frame tolerance.')
        elif not np.any(SimpleITK.GetArrayFromImage(gt)):
            # If the ground truth is empty for the selected frame,
            # find next optimal frame in the ground truth, within the same sweep
            sweep_index = find_sweep_index(frame)
            nearest_frame = find_closest_annotated_frame(
                gt_array, sweep_index, frame)
            # print('Ground truth is empty. Nearest annotated frame:', nearest_frame)

            # If there is no next optimal frame, set the Hausdorff distance to maximum sweep width (744)
            if nearest_frame is None:
                hausdorff_distance = 744 * MAX_FRAME_TOLERANCE
                # print('No nearest optimal frame found. Setting Hausdorff distance to maximum sweep width (744) scaled by the frame tolerance.')

            else:
                gt_nearest_frame = gt_array[nearest_frame].copy()
                gt_nearest_frame[gt_nearest_frame > 0] = 1
                gt_nearest_frame = SimpleITK.GetImageFromArray(
                    gt_nearest_frame)
                gt_nearest_frame = caster.Execute(gt_nearest_frame)

                # Compute the distance to the next optimal frame
                hausdorff_calculator.Execute(gt_nearest_frame, pred)
                hausdorff_distance = hausdorff_calculator.GetHausdorffDistance() * \
                    (nearest_frame - frame)

                # Compute dice coefficient for the nearest annotated frame
                overlap_measures.Execute(gt_nearest_frame, pred)
                dice_coefficient_nearest_frame = overlap_measures.GetDiceCoefficient()
                # Adjust Dice score with a coefficient based on the distance between frames
                distance_coefficient = 1 - \
                    abs(nearest_frame - frame) / MAX_FRAME_TOLERANCE
                dice_coefficient_soft = dice_coefficient_nearest_frame * distance_coefficient
                metrics['DiceCoefficientNearestFrame'] = dice_coefficient_nearest_frame
                metrics['NearestAnnotatedFrame'] = nearest_frame
                metrics['DiceCoefficientSoft'] = dice_coefficient_soft

        metrics['HausdorffDistance'] = hausdorff_distance
        return metrics


def evaluate_selected_frame(masks, frame_number):
    """
    Check for segmentations in a given frame.

    Parameters:
    masks (ndarray): A 3D array where the first dimension is the frame number and the next two are the spatial dimensions.
    frame_number (int): The specific frame number to check in the masks array.

    Returns:
    tuple: A tuple containing two booleans. The first boolean indicates if there is any segmentation with a value of 1, 
    and the second boolean is True if there is any segmentation with a value of 2.
    """
    # Check if the frame_number is within the range of available frames
    if frame_number < 0 or frame_number >= masks.shape[0]:
        raise ValueError("Frame number is out of range")

    # Access the specific frame
    frame = masks[frame_number]

    # Check if there is any segmentation with a value of 1
    has_1 = np.any(frame == 1)

    # Check if there is any segmentation with a value of 2
    has_2 = np.any(frame == 2)

    return has_1, has_2


def compute_wfss(gt_masks, selected_frame_number):
    """
    Compute the Weighted Frame Selection Score (WFSS) for the selected frame.

    Parameters:
    gt_masks (ndarray): A 3D array where the first dimension is the frame number and the next two are the spatial dimensions, containing segmentation values.
    selected_frame_number (int): The frame number selected by the algorithm.

    Returns:
    float: The WFSS score for the selected frame.
    """

    # If there are no segmentations at all in the dataset, return 0 as we cannot evaluate the frames.
    if not np.any(gt_masks > 0):
        return 0

    # If no frame was selected (= -1), because from previous if statement we know there is at least
    # one annotation in the ground-truth, return 0.
    if selected_frame_number == -1:
        return 0

    # Evaluate the selected frame to determine if it contains optimal or suboptimal planes
    has_optimal, has_suboptimal = evaluate_selected_frame(
        gt_masks, selected_frame_number)

    # If no optimal planes are available in the entire dataset, and the algorithm selects a suboptimal one, it gets a full score.
    if 1 not in np.unique(gt_masks) and has_suboptimal:
        return 1

    # If an optimal plane is selected, the algorithm receives a full score.
    if has_optimal:
        return 1

    # If a suboptimal plane is selected when an optimal one is available, it receives a partial score.
    if not has_optimal and has_suboptimal and 1 in np.unique(gt_masks):
        return 0.6

    # If the algorithm selects a frame without any optimal or suboptimal planes while such planes exist, it receives the minimum score.
    return 0


def calculate_ellipse_circumference_mm(segmentation_mask, pixel_spacing):
    """
    Fit ellipses to the segmentation mask, calculate the circumferences,
    find the largest circumference, and convert it to millimeters.

    Args:
    segmentation_mask (np.array): The segmentation mask where ellipses are to be fitted.
    pixel_spacing (float): The pixel spacing to convert pixel measurements to millimeters.

    Returns:
    float: The largest circumference in millimeters.
    """
    _, circumference, _ = fit_ellipses(segmentation_mask)

    if circumference is None:
        return None  # Handle cases with no circumferences

    # Convert the largest circumference from pixels to millimeters
    circumference_mm = pixels_to_mm(
        circumference, pixel_spacing)

    return circumference_mm


def find_sweep_index(fetal_abdomen_frame_number):
    # Fixed constant sweep indices
    sweep_indices = {1: (0, 140), 2: (140, 280), 3: (
        280, 420), 4: (420, 560), 5: (560, 700), 6: (700, 840)}

    # Check which sweep index the frame number belongs to
    for sweep_index, (start, end) in sweep_indices.items():
        if start <= fetal_abdomen_frame_number < end:
            return sweep_index
    return None  # Return None if the frame number doesn't belong to any sweep


def find_closest_annotated_frame(gt_array, sweep_index, frame):
    # Define the sweep boundaries
    sweep_indices = {1: (0, 140), 2: (140, 280), 3: (280, 420),
                     4: (420, 560), 5: (560, 700), 6: (700, 840)}

    # Get the start and end frame indices for the current sweep
    start_index, end_index = sweep_indices[sweep_index]

    # Calculate actual search boundaries considering the frame tolerance
    search_start = max(start_index, frame - MAX_FRAME_TOLERANCE)
    search_end = min(end_index, frame + MAX_FRAME_TOLERANCE + 1)

    # Initialize variables to store the closest next and previous annotated frames
    next_annotated_frame = None
    previous_annotated_frame = None

    # Search forward wuthin the frame tolerance
    for next_frame in range(frame + 1, search_end):
        if np.any(gt_array[next_frame]):
            next_annotated_frame = next_frame
            break

    # Search backward within the frame tolerance
    for prev_frame in range(frame - 1, search_start - 1, -1):
        if np.any(gt_array[prev_frame]):
            previous_annotated_frame = prev_frame
            break

    # Decide which annotated frame is closest
    if previous_annotated_frame is None and next_annotated_frame is None:
        return None
    elif previous_annotated_frame is None:
        return next_annotated_frame
    else:
        return previous_annotated_frame
