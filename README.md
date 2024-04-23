# ACOUSLIC-AI-evaluation-method
This repository contains the evaluation software used in the [ACOUSLIC-AI challenge](https://acouslic-ai.grand-challenge.org/), which focuses on the segmentation of fetal abdomen in 2D prenatal abdominal
ultrasound sequences. These sequences were acquired following a standardized Obstetric Sweep Protocol (OSP) proposed by [DeStigter et al (2011)](https://doi.org/10.1109/GHTC.2011.39). It is a blind sweep acquisition protocol consisting of six sweeps
over the gravid abdomen. The primary objective of the challenge is to estimate abdominal circumference (AC) measurements from OSP data acquired by novice operators in five African peripheral healthcare units and one European
hospital. 

## Managed by
[Diagnostic Image Analysis Group](https://diagnijmegen.nl/) and [Medical UltraSound Imaging Center](https://music.radboudimaging.nl/), Radboud University Medical Center, Nijmegen, the Netherlands.

## Contact information
Sofía Sappia: mariasofia.sappia@radboudumc.nl \
Keelin Murphy: keelin.murphy@radboudumc.nl

## Ground truth
The evaluation process is designed to rigorously assess the performance of submitted algorithms against a ground truth dataset, which comprises:
- **Fetal abdomen stacked masks:** `.mha` file containing a stack of fetal abdomen masks delineated as 'optimal' (pixel values equal to 1) or 'suboptimal' (pixel values equal to 2) for AC estimation. 
- **Fetal abdominal circumferences:** reference fetal AC measuerements (in mm) derived from the best available — optimal, otherwise suboptimal — annotation masks on each of the six sweeps, wherever such annotations were present. These reference measurements serve as a standard for evaluating the accuracy of the algorithms' estimations.

## Expected algorithm outputs
Algorithms evaluated in this software package are expected to provide the following outputs:
- **Fetal abdomen segmentation mask:** a single 2D segmentation mask of the fetal abdomen where 0-background and 1-fetal abdomen. This segmentation mask should be fit for the [ellipse fitting tool](src/acouslicaieval/ellipse_fitting.py) to measure its circumference during evaluation. Pixels extending beyond the field of view (FOV) of the ultrasound beam are not considered during evaluation.
- **Fetal abdomen ultrasound frame number:** Integer that represents the fetal abdomen ultrasound frame number corresponding to the output segmentation mask, indexed from zero. The value can also be -1, indicating that no frame was selected.

## Evaluation Metrics
The [ACOUSLIC-AI challenge](https://acouslic-ai.grand-challenge.org/) employs a comprehensive suite of metrics to evaluate the performance of the participating algorithms:

- **Dice Similarity Coefficient (DSC):** This metric quantifies the spatial overlap accuracy of the algorithm's segmentation against the ground truth mask. A higher DSC indicates a closer match to the ground truth and thus a better segmentation performance. It's important to note that the ground truth mask, if available, corresponds to the annotation in the specified frame of the fetal abdomen stack (i.e., this metric is computed on the 2D ground truth and prediction masks corresponding to the fetal frame number). For this comparison, the ground truth mask is converted to a binary format (1 representing the fetal abdomen and 0 representing the background). 
- **Weighted Frame Selection Score (WFSS):** WFSS evaluates the algorithm's frame selection accuracy, assigning higher scores to accurately identified and chosen clinically relevant frames. A score of 1 denotes correct identification of optimal planes, 0.6 for suboptimal plane selection when an optimal is available, and 0 for the selection of irrelevant frames when optimal/suboptimal ones are present.
- **Hausdorff Distance (HD):** This metric measures the maximum distance between the algorithm's predicted fetal adomen mask boundary and the actual ground truth boundary in the selected frame, providing a sense of the largest potential error in the segmentation boundary prediction. Similarly to the computation of the DICE coefficient, the 2D ground truth mask in the selected frame is converted to a binary format for evaluation against the 2D predicted mask. Additionally, only the pixels within the field of view of the ultrasound beam are considered during this process.
- **Normalized Absolute Error (NAE):** the normalized absolute error for abdominal circumference measurements provides a scale-independent measure of the precision in abdominal circumference estimation. It's calculated by taking the absolute difference between the ground truth and the predicted circumference, normalized by the maximum of either value to account for the scale:

  \[ \text{NAE}_{\text{AC}} = \frac{|\text{AC}_{\text{gt}} - \text{AC}_{\text{pred}}|}{\max(\text{AC}_{\text{gt}}, \text{AC}_{\text{pred}}, \epsilon)} \]

  Where:
  - \( \text{NAE}_{\text{AC}} \) is the Normalized Absolute Error for Abdominal Circumference.
  - \( \text{AC}_{\text{gt}} \) is the ground truth Abdominal Circumference measurement — if present — in the sweep corresponding to the algorithm's selected frame.
  - \( \text{AC}_{\text{pred}} \) is the algorithm's predicted Abdominal Circumference measurement.
  - \( \epsilon \) is a small constant to prevent division by zero, set to \( 1 \times 10^{-6} \).
  
  A lower NAE indicates a higher accuracy in predicting the AC measurements from the segmented masks, which is crucial for clinical applicability. \
  **Note:** The predicted abdominal circumference used to compute this metric is measured using the `fit_ellipses` function in the [ellipse fitting tool](src/acouslicaieval/ellipse_fitting.py) provided in this repository. Ellipses extending beyond the field of view of the ultrasound beam are extrapollated using the contour points contained within the FOV.

The combined use of these metrics allows for a balanced evaluation of the algorithms, not only in terms of their segmentation accuracy but also their practical utility in a clinical setting.

## Execution Environment
The evaluation is performed in a containerized environment to ensure consistency. To run the evaluation locally, use the following command:

    ./test_run.sh

This will process data from ./test/input, compute the evaluation metrics, and save the results to ./test/output.

## Docker Container Packaging
For deployment on Grand-Challenge.org, the evaluation method is packaged into a Docker container using the following command:

    docker save acouslicai-evaluation-method | gzip -c > acouslicai-evaluation-method.tar.gz

## Repository Structure
Below is a description of the key directories and files within this repository:

- `data/`: Contains the mask corresponding to the field of view of the ultrasound beam.
- `ground_truth/`: Contains the ground truth masks for the example test segmentations in `test/`.
- `src/`: Source code for the `acouslicaieval` package, which contains evaluation metrics computation.
- `test/`: Contains sample input/output data for local testing.
- `evaluate.py`: This is the main evaluation script. It performs several key functions:
    - Loads the predicted segmentation masks: The script begins by importing the masks generated by the algorithms.
    - Fits an ellipse to the segmentation mask: Once the mask is loaded, the script applies an ellipse fitting process to estimate the abdominal circumference.
    - Measures the ellipse’s circumference: After fitting the ellipse, the script calculates its circumference.
    - Compares the predicted data to the ground truth: The script evaluates the predicted masks and the circumference measurements against the ground truth data. This includes comparing the segmentation accuracy of the abdomen, the classification accuracy of the selected frames, and the estimation error in the AC measurements.

## License
This project is licensed under the [LICENSE](https://github.com/DIAGNijmegen/ACOUSLIC-AI-evaluation-method/blob/main/LICENSE) file contained in the repository.