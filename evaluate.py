"""
This script evaluates segmentation results for fetal abdomen images in the ACOUSLIC-AI challenge. 
It loads the segmentation masks provided by the algorithm, compares them to the ground truth masks,
and computes various metrics including overlap metrics, weighted frame selection score (WFSS),
and normalized absolute error (NAE) for the circumference measurements.

To execute, this script is typically run within a containerized environment. For local execution,
you can trigger the evaluation using a provided bash script:

    ./test_run.sh

This script processes input data from './test/input', computes evaluation metrics,
and saves the results in './test/output'.

To prepare this script for deployment on Grand-Challenge.org, you would typically package it into
a Docker container and save it using:

    docker save acouslicai-evaluation-method | gzip -c > acouslicai-evaluation-method.tar.gz
"""
import json
from glob import glob
from multiprocessing import Pool
from pathlib import Path
from pprint import pformat, pprint

import acouslicaieval.evaluation as eval
import numpy as np
import SimpleITK

# Define input and output directories
INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")
GROUND_TRUTH_DIRECTORY = Path("ground_truth")

# Pixel spacing is defined for converting pixel measurements to millimeters
PIXEL_SPACING = 0.28


def main():
    print_inputs()

    metrics = {}
    predictions = read_predictions()

    # We now process each algorithm job for this submission
    # Note that the jobs are not in any order!
    # We work that out from predictions.json

    # Start a number of process workers, using multiprocessing
    # The optimal number of workers ultimately depends on how many
    # resources each process() would call upon
    with Pool(processes=4) as pool:
        metrics["results"] = pool.map(process, predictions)

    # Now generate the overall scores for this submission
    metrics = metrics["results"]
    metrics = aggregate_metrics(metrics)

    # Create score = 0.5 * nae_circumference + 0.25 * DiceCoefficient + 0.25 * wfss
    metrics["score"] = {"mean":
                        0.5 * metrics["nae_circumference"]["mean"] +
                        0.25 * metrics["DiceCoefficient"]["mean"] +
                        0.25 * metrics["wfss"]["mean"],
                        "std": 0.5 * metrics["nae_circumference"]["std"] +
                        0.25 * metrics["DiceCoefficient"]["std"] +
                        0.25 * metrics["wfss"]["std"]
                        }

    # Make sure to save the metrics
    write_metrics(metrics=metrics)

    return 0


def process(job):
    """
    Processes a single algorithm job by analyzing the outputs and computing various evaluation metrics.

    Args:
    job (dict): A dictionary containing job details, including input and output information.

    Returns:
    dict: A dictionary containing computed metrics for the given job.
    """

    # Begin processing and log the job details for reference.
    report = "Processing:\n"
    report += pformat(job)
    report += "\n"

    # 1. Determine the location of the segmentation and frame number results
    fetal_abdomen_segmentation_location = get_file_location(
        job_pk=job["pk"],
        values=job["outputs"],
        slug="fetal-abdomen-segmentation",
    )
    fetal_abdomen_frame_number_location = get_file_location(
        job_pk=job["pk"],
        values=job["outputs"],
        slug="fetal-abdomen-frame-number",
    )

    # 2. Load the segmentation results and the selected frame number from the algorithm's outputs
    fetal_abdomen_segmentation = load_image_file(
        location=fetal_abdomen_segmentation_location,
    )
    # assert it is a binary mask
    assert np.all(np.isin(np.unique(fetal_abdomen_segmentation),
                  [0, 1])), "Segmentation mask should be binary."

    fetal_abdomen_frame_number = load_json_file(
        location=fetal_abdomen_frame_number_location,
    )

    # get the associated sweep number, if any
    sweep_index = eval.find_sweep_index(fetal_abdomen_frame_number)

    # Get the mask corresponding to the selected frame number (we saved it as 3d for visualization purposes)
    if sweep_index is not None and fetal_abdomen_frame_number != -1:
        fetal_abdomen_segmentation = fetal_abdomen_segmentation[fetal_abdomen_frame_number]
    else:  # return empty mask if no (valid) frame number is provided
        fetal_abdomen_segmentation = np.zeros_like(
            fetal_abdomen_segmentation[0])

    # 3. Retrieve the input image name to match it with an image and reference measurement in the ground truth
    stacked_fetal_ultrasound_image_name = get_image_name(
        values=job["inputs"],
        slug="stacked-fetal-ultrasound",
    )

    # 4. Load the ground truth segmentation and circumference measurement
    fetal_abdomen_segmentation_gt, fetal_abdominal_circumference_gt = load_ground_truth(
        file_name=stacked_fetal_ultrasound_image_name, sweep_index=sweep_index)

    # 5. Calculate the circumference from the segmentation for the specified frame and convert it to millimeters.
    # In this step we first calculate the circumference of the ellipse that fits the segmentation mask.
    fetal_abdominal_circumference_pred = eval.calculate_ellipse_circumference_mm(
        fetal_abdomen_segmentation, PIXEL_SPACING)

    # 6. Compute the normalized absolute error (NAE) of the circumference
    # Set the nae_circumference to 1 (worst score) if no predicted circumference or ground truth are available
    if fetal_abdominal_circumference_pred is None or fetal_abdominal_circumference_gt is None:
        nae_circumference = 1
        # Log a warning if no valid circumference could be calculated.
        if fetal_abdominal_circumference_pred is None:
            print("No valid circumferences calculated.")
    else:
        # Calculate the normalized absolute error (NAE) of the circumference
        ae_circumference = np.abs(
            fetal_abdominal_circumference_gt - fetal_abdominal_circumference_pred)
        nae_circumference = ae_circumference / \
            max(fetal_abdominal_circumference_gt,
                fetal_abdominal_circumference_pred, 1e-6)

    # 7. Compute overlap metrics using the provided segmentation and the ground truth
    metrics_overlap = eval.FetalAbdomenSegmentationEval().score_case(gt_array=fetal_abdomen_segmentation_gt,
                                                                     pred=fetal_abdomen_segmentation, frame=fetal_abdomen_frame_number)

    # 8. Calculate the Weighted Frame Selection Score (WFSS) based on the ground truth and the selected frame
    wfss = eval.compute_wfss(fetal_abdomen_segmentation_gt,
                             fetal_abdomen_frame_number)

    # Compile all computed metrics into a single dictionary for output
    all_metrics = dict(metrics_overlap)
    all_metrics.update({
        "wfss": wfss,
        "nae_circumference": nae_circumference
    })

    # Print and log the computed metrics for the current job
    print('#'*100)
    print('Metrics for file:', stacked_fetal_ultrasound_image_name)
    pprint(all_metrics)
    print(report)
    print('#'*100)

    return all_metrics


def aggregate_metrics(results, specific_metrics=None):
    """
    Aggregates metrics from multiple result dictionaries.

    Args:
        results (list of dict): A list containing dictionaries of metrics for each result.
        specific_metrics (list of str, optional): List of specific metrics to aggregate. If None, aggregates all metrics.

    Returns:
        dict: A dictionary containing the mean of each metric across all results.
    """
    # Compile all computed metrics from the results into a single dictionary for output
    all_metrics = {}
    for result in results:
        for key, value in result.items():
            if specific_metrics is None or key in specific_metrics:
                if key in all_metrics:
                    all_metrics[key].append(value)
                else:
                    all_metrics[key] = [value]

    # Calculate mean and standard deviation of each metric to aggregate them
    aggregated_metrics = {}
    for metric, values in all_metrics.items():
        mean_value = np.mean(values)
        std_value = np.std(values)
        aggregated_metrics[metric] = {"mean": mean_value, "std": std_value}

    return aggregated_metrics


def print_inputs():
    # Just for convenience, in the logs you can then see what files you have to work with
    input_files = [str(x)
                   for x in Path(INPUT_DIRECTORY).rglob("*") if x.is_file()]

    print("Input Files:")
    pprint(input_files)
    print("")


def read_predictions():
    # The prediction file tells us the location of the users' predictions
    with open(INPUT_DIRECTORY / "predictions.json") as f:
        return json.loads(f.read())


def get_image_name(*, values, slug):
    # This tells us the user-provided name of the input or output image
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def get_interface_relative_path(*, values, slug):
    # Gets the location of the interface relative to the input or output
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["interface"]["relative_path"]

    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk, values, slug):
    # Where a job's output file will be located in the evaluation container
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_json_file(*, location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


def load_ground_truth(*, file_name, sweep_index=None):
    uuid = Path(file_name).stem
    # Reads an mha file
    img_itk = SimpleITK.ReadImage(
        GROUND_TRUTH_DIRECTORY / uuid / "fetal_abdomen_stacked_masks" / file_name)
    img_np = SimpleITK.GetArrayFromImage(img_itk)

    # Load the json file with the ground truth measurements
    with open(GROUND_TRUTH_DIRECTORY / uuid / "fetal_abdominal_circumference.json") as f:
        circumference_mm = json.load(f)
    if sweep_index is not None:
        circumference_mm = circumference_mm[f"sweep_{sweep_index}_ac_mm"]
    else:
        circumference_mm = None
    return img_np, circumference_mm


def load_image_file(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tiff")) + \
        glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def write_metrics(*, metrics):
    # Write a json document used for ranking results on the leaderboard
    with open(OUTPUT_DIRECTORY / "metrics.json", "w") as f:
        f.write(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    raise SystemExit(main())
