## Detection Limits of Artificial Intelligence (AI) based Object Detection from Scanning Electron Microscopy Images

## Software or Data description
   - Statements of purpose and maturity
   - Description of the repository contents
   - Technical installation instructions, including operating
     system or software dependencies 

## Workflow of Computations
- Step 1: compute data quality metrics using generate_metrics.py
- Step 2: plot data quality metrics as a function of contrast and noise using plot_image_quality.py
- Step 3: train UNet model on set 1 - set 5 (Web Image Processing Workflow)
- Step 4: infer image masks for set 6 using the trained UNet model and evaluate its accuracy  (Web Image Processing Workflow)
- Step 5: merge the data quality metrics and AI model accuracy metrics using match_ai_data.py
- Step 6: plot relationships between data quality metrics and AI model accuracy metrics using plot_ai_model_predictions.py

## Contact information
   - Peter Bajcsy, ITL NIST, Software and Systems Division, Information Systems Group
   - Contact email address at NIST: peter dot bajcsy at nist dot gov

## Citation of the work
   - Peter Bajcsy, Brycie Wiseman, Michael Majurski, and Andras E. Vladar, "Detection Limits of AI-based SEM Dimensional Metrology", Proceedings of SPIE conference on Advanced Lithography + Patterning, 23 - 27 February 2025, San Jose, California, US, [URL](https://spie.org/conferences-and-exhibitions/advanced-lithography-and-patterning/program)

## LICENSE
- The version of [LICENSE.md](LICENSE.md) included in this
  repository is approved for use.
- Updated language on the [Licensing Statement][nist-open] page
  supersedes the copy in this repository. You may transcribe the
  language from the appropriate "blue box" on that page into your
  README.


## Related material
   - We used ARTIMAGEN SEM Simulation Software to generate images with varying contrast and noise l evel.
     - Cizmar P., Vladár A., Postek M. “Optimization of accurate SEM imaging by use of artificial images”, Proc. SPIE 7378, Scanning Microscopy, 737815, 2009, [URL](https://doi.org/10.1117/12.823415)
     - [Project URL](https://sourceforge.net/projects/artimagen/) and [GitHub Repo URL](https://github.com/strec007/artimagen)
     - License: As this software was developed as part of work done by the United States Government, it is not subject to copyright, and is in the public domain. Note that according to GNU.org public domain is compatible with GPL.

   - We used UNet Convolutional Neural Network (CNN) AI model implementation By Michael Majurski (NIST) for training and inference of image segmentation
     - Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.
     - Name: WIPP UNet CNN Training Plugin and WIPP UNet CNN Inference Plugin 
     - Title: WIPP UNet CNN Training Plugin, Version: 1.0.0, [Repository](https://github.com/usnistgov/WIPP-unet-train-plugin), Container image: wipp/wipp-unet-cnn-train-plugin:1.0.0 
     - Title: WIPP UNet CNN Inference Plugin, Version:1.0.0, [Repository](https://github.com/usnistgov/WIPP-unet-inference-plugin) Container images: wipp/wipp-unet-cnn-inference-plugin:1.0.0

## CODEOWNERS
The file named
[CODEOWNERS](CODEOWNERS) can be viewed to discover
which GitHub users are "in charge" of the repository. More
crucially, GitHub uses it to assign reviewers on pull requests.
GitHub documents the file (and how to write one) [here][gh-cdo].


## CODEMETA
Project metadata is captured in `CODEMETA.yaml`, used by the NIST
Software Portal to sort the GitHub work under the appropriate thematic
homepage.
