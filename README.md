[![License](https://img.shields.io/badge/license-NIST--Software-blue.svg)](https://github.com/usnistgov/detection_limits/blob/main/LICENSE)

# Detection Limits of Artificial Intelligence (AI) based Object Detection from Scanning Electron Microscopy (SEM) Images

This project is about designing a methodology for quantifying and relating detection limits of AI model-based measurements
from SEM images to digital image quality metrics and to human observers.

## Statements of purpose and maturity

This software supports the methodology for quantifying detection limits for AI-based measurements from SEM images.
The detection limits establish the relationship between the quality of SEM images and human or AI model detection performance
The software is actively developed.

## Description of the repository contents

The repository contains the software for

- extracting image quality metrics from simulated SEM image collections
- merging image quality metrics with AI model accuracy metrics, where the AI model was trained on the same simulated SEM image collection
- plotting image quality metrics and AI model accuracy metrics as a function of SEM Image simulation noise and contrast parameters
- plotting relationships between detection limits of human (eye model) and numerical (AI model) observers
- plotting relationships between AI model single-valued metrics (Dice, FNR, FPR) and multiple SNR definitions
- interactively interrogating all plots via web interface

The repository contains two folders:

- src folder: contains all Python scripts
- web folder: contains all HTML, JavaScript, and CSS files together with the plots

## Installation

To install the package, simply run:

```console
pip install detection-limits
```

## Usage

You can use the package to calculate image quality metrics for SEM images.

### 1. Import the package

```python
import detection_limits as dl
import numpy as np
```

### 2. Calculate metrics for a single image

```python
# Load your image and mask (example using numpy arrays)
image = np.random.rand(1024, 1024)
mask = np.random.randint(0, 2, (1024, 1024))

# Calculate metrics
metrics = dl.calculate_all_metrics(image, mask)
print(metrics)
```

### 3. Calculate metrics for a batch of images

```python
# Define input and output paths
input_intensity_path = "./path/to/images"
input_mask_path = "./path/to/masks"
output_csv_path = "./results.csv"

# Run metrics calculation
dl.metrics(input_intensity_path, input_mask_path, output_csv_path, set_index=1)
```

## Future Work

Future updates will include:

- Computational efficiency improvements
- Automated report generation
- GPU support for faster processing

## Workflow of Computations

- Step 1: compute data quality metrics using `dl-metrics`
- Step 2: plot data quality metrics as a function of contrast and noise using `dl-plot-quality`
- Step 3: train UNet model on set 1 - set 5 (Web Image Processing Workflow)
- Step 4: infer image masks for set 6 using the trained UNet model and evaluate its accuracy  (Web Image Processing Workflow)
- Step 5: merge the data quality metrics and AI model accuracy metrics using `dl-match`
- Step 6: plot relationships between data quality metrics and AI model accuracy metrics using `dl-plot-ai`
- Step 7: support decisions to obtain a trusted AI-based measurement by applying an AI model with a user-defined minimum accuracy requirement to an input SEM image with minimum SNR characteristics defined by the graph generated using `dl-analyze`

## Contact information

- Peter Bajcsy, ITL NIST, Software and Systems Division, Information Systems Group
- Contact email address at NIST: peter dot bajcsy at nist dot gov

## Citation of the work

- Peter Bajcsy, Brycie Wiseman, Michael Majurski, and Andras E. Vladar, "Detection Limits of AI-based SEM Dimensional Metrology", Proceedings of SPIE conference on Advanced Lithography + Patterning, 23 - 27 February 2025, San Jose, California, US, [URL](https://spie.org/conferences-and-exhibitions/advanced-lithography-and-patterning/program)
- Peter Bajcsy, Pushkar Sathe, and Andras E. Vladar, "Relating human and AI-based detection limits in SEM dimensional metrology", Under review.

## LICENSE

- The version of [LICENSE](https://github.com/usnistgov/detection_limits/blob/main/LICENSE) included in this
  repository is approved for use.
- Updated language on the [Licensing Statement][nist-open] page
  supersedes the copy in this repository. You may transcribe the
  language from the appropriate "blue box" on that page into your
  README.

## Related material

- We used ARTIMAGEN SEM Simulation Software to generate images with varying contrast and noise level.
  - Cizmar P., Vladár A., Postek M. “Optimization of accurate SEM imaging by use of artificial images”, Proc. SPIE 7378, Scanning Microscopy, 737815, 2009, [URL](https://doi.org/10.1117/12.823415)
  - [Project URL](https://sourceforge.net/projects/artimagen/) and [GitHub Repo URL](https://github.com/strec007/artimagen)
  - License: As this software was developed as part of work done by the United States Government, it is not subject to copyright, and is in the public domain. Note that according to GNU.org public domain is compatible with GPL.

- We used UNet Convolutional Neural Network (CNN) AI model implementation By Michael Majurski (NIST) for training and inference of image segmentation
  - Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.
  - Name: WIPP UNet CNN Training Plugin and WIPP UNet CNN Inference Plugin
  - Title: WIPP UNet CNN Training Plugin, Version: 1.0.0, [Repository](https://github.com/usnistgov/WIPP-unet-train-plugin), Container image: wipp/wipp-unet-cnn-train-plugin:1.0.0
  - Title: WIPP UNet CNN Inference Plugin, Version:1.0.0, [Repository](https://github.com/usnistgov/WIPP-unet-inference-plugin) Container images: wipp/wipp-unet-cnn-inference-plugin:1.0.0

- The execution and full computational provenance were obtained by using the WIPP scientific workflow software:
  - [GitHub: WIPP code](https://github.com/usnistgov/WIPP)
  - Bajcsy, P. , Chalfoun, J. and Simon, M. (2018), Web Microanalysis of Big Image Data, Springer International Publishing.

## Acknowledgement

- This work was performed with funding from the CHIPS Metrology Program, part of CHIPS for America, National Institute of Standards and Technology, U.S. Department of Commerce.

## CODEOWNERS

The file named
[CODEOWNERS](https://github.com/usnistgov/detection_limits/blob/main/CODEOWNERS) can be viewed to discover
which GitHub users are "in charge" of the repository. More
crucially, GitHub uses it to assign reviewers on pull requests.
GitHub documents the file (and how to write one) [here][gh-cdo].

## CODEMETA

Project metadata is captured in `CODEMETA.yaml`, used by the NIST
Software Portal to sort the GitHub work under the appropriate thematic
homepage.
