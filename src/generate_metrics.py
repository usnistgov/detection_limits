'''
NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
'''

# Author: Peter Bajcsy and Pushkar Sathe
# Date: 2025-04-01
# Description: This script computes image quality metrics from SEM images

import os
import numpy as np
from skimage import io
import sys
import scipy.stats
import re

import cv2
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from skimage import feature
import pandas as pd

import time
import argparse
import warnings


'''
compute image SNR quality metrics based on the intensity and mask pairs of images
'''
def calculate_all_snr_with_mask(
    image: np.ndarray, mask: np.ndarray, noise=None
) -> tuple[float, float, float, float, float, float, float, float, float, float,float, float, float, float]:
    # Normalize the image first
    # normalized_image = normalize_image(image)
    normalized_image = image.copy()
    foreground_region = normalized_image[mask > 0]
    background_region = normalized_image[mask == 0]

    foreground_mean = foreground_region.flatten().mean()
    foreground_var = foreground_region.flatten().var()
    foreground_std = foreground_region.flatten().std()

    background_mean = background_region.flatten().mean()
    background_var = background_region.flatten().var()
    background_std = background_region.flatten().std()

    # foreground_mean = np.mean(foreground_region)
    # foreground_var = np.var(foreground_region)
    # foreground_std = np.std(foreground_region)

    # background_mean = np.mean(background_region)
    # background_var = np.var(background_region)
    # background_std = np.std(background_region)

    snr1 = (
        foreground_var / background_var if background_var != 0 else 0
    )  # using power definition E(Signal^2) / E(Noise^2)
    snr2 = (
        np.sqrt(foreground_mean) / np.sqrt(foreground_var) if foreground_var != 0 else 0
    ) # using root mean square of E(Signal^2] /E{Noise^2] SNR definition for harmonic signals
    snr3 = (
        (foreground_mean * foreground_mean) / (background_std * background_std) if background_mean != 0 else 0
    )  # using sensitivity index for films definition of SNR with estimated background standard deviation
    snr4 = (
        foreground_mean / background_std if background_std != 0 else 0
    ) # using 1 over coefficient of variation & estimated background standard deviation
    snr5 = (
        foreground_mean / noise if noise != 0 else 0
    ) # using 1 over coefficient of variation & simulated (known) noise parameter
    snr6 = (
        (foreground_mean*foreground_mean) / (noise * noise) if noise != 0 else 0
    ) # using sensitivity index for films definition of SNR with simulated (known) noise parameter
    snr7 = (
        foreground_var / (noise * noise) if noise != 0 else 0
    )  # using power definition E(Signal^2) / E(Noise^2) and simulated (known) noise parameter
    snr8 = (
        np.sqrt(foreground_mean) / noise if noise != 0 else 0
    ) # using root mean square of E(Signal^2] /E{Noise^2] SNR definition for harmonic signals and simulated (known) noise parameter
    snr9 = (
        (foreground_mean - background_mean) / background_std if background_std != 0 else 0
    ) # using Cohen's d index  and estimated background standard deviation: https://en.wikipedia.org/wiki/Effect_size#Cohen's_d
    # Cohen's d is frequently used in estimating sample sizes for statistical testing.
    snr10 = (
        (foreground_mean - background_mean) / noise if noise != 0 else 0
    ) # using Cohen's d index  and  simulated (known) noise parameter

    # sensitivity index for films definition of SNR with estimated background standard deviation
    # print(f"foreground_mean: {foreground_mean},\t background_mean: {background_mean},\t foreground_std: {foreground_var},background_std: {background_var},\t snr: {snr}, snr2: {snr2}")
    return (
        float(snr1),
        float(snr2),
        float(snr3),
        float(snr4),
        float(snr5),
        float(snr6),
        float(snr7),
        float(snr8),
        float(snr9),
        float(snr10),
        float(foreground_mean),
        float(background_mean),
        float(foreground_var),
        float(background_var)
    )

'''
This method computes image quality metrics from loaded image, mask, and reference intensity image
calculate_metrics(img, mask, reference_image, noise_val, contrast_val, file name, set index)
'''
def calculate_all_metrics(
    image, mask=None, reference_image=None, noise_val=None, contrast_val=None, intensityimage_filename="test.tif", set_index =1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress all warning types
        """Calculates various image quality metrics."""
        start_time = time.time()
        image_flat = image.flatten()
        reference_flat = None
        mask_flat = None

        #############################################
        # single image metrics
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        variance_intensity = np.var(image)

        michelson_contrast = (
            (np.max(image) - np.min(image)) / (np.max(image) + np.min(image))
            if np.abs(np.max(image) + np.min(image)) > 0.0001
            else 0
        )
        # https://en.wikipedia.org/wiki/Contrast_(vision)
        # The RMS contrast is found by calculating the standard deviation of the pixel intensities
        rms_contrast = std_intensity

        # edge density metric
        edges = feature.canny(image)
        edge_density = np.sum(edges) / image.size
        print("DEBUG:  np.sum(edges)=",  np.sum(edges))
        # time2 = time.time() - time1

        #############################################
        # pair-wise image metrics with reference intensity image
        if reference_image is not None:
            #reference_flat = reference_image.flatten()
            ssim_score = ssim(
                image, reference_image, data_range=image.max() - image.min()
            )
            psnr = cv2.PSNR(image, reference_image)  # OpenCV's PSNR calculation
        else:
            smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)
            ssim_score = ssim(
                image, smoothed_image, data_range=image.max() - image.min()
            )
            psnr = cv2.PSNR(image, smoothed_image)

        #############################################
        # pair-wise image metrics with reference segmentation mask image
        if mask is not None:
            mask_flat = mask.flatten()
            # scikit-learn implementation
            mi = mutual_info_score(image_flat, mask_flat)
            # Calculate normalized mutual information (NMI)
            # This is a more robust measure than MI as it's normalized
            # to account for the entropy of each variable
            nmi = normalized_mutual_info_score(image_flat, mask_flat)
            # time3 = time.time() - time2

            # compute the cress-entropy of the image
            # https://en.wikipedia.org/wiki/Entropy_(information_theory)
            # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
            ce = compute_image_entropy(image)
        else:
            mi = None
            ce = None
            nmi = None
        noise = float(noise_val)
        #print(f"noise: {noise}, type: {type(noise)}")
        (
            snr1,
            snr2,
            snr3,
            snr4,
            snr5,
            snr6,
            snr7,
            snr8,
            snr9,
            snr10,
            foreground_mean,
            background_mean,
            foreground_var,
            background_var
        ) = calculate_all_snr_with_mask(image, mask, noise=noise)
        time4 = time.time() - start_time
        # if q:
        # q.put(f"start: {start_time}, time1: {time1}, time2: {time2}, time3: {time3}, time4: {time4}")
        #print(f"total time: {time4}")
        # logging.info()

    # add .ome to the file name for the image since the WIPP-based UNet AI models were trained on OME.TIF files instead of the original
    # filenames without OME and with .tiff extension.
    # this is added in order to support automated merging of AI model and Data Quality metrics
    intensityimage_ome_filename = os.path.splitext(intensityimage_filename)[0] + ".ome.tif"
    #print("DEBUG: ", intensityimage_ome_filename)
    return {
        "IMAGE-NAME": intensityimage_ome_filename,
        "Set_index": set_index,
        "Noise_level": noise_val,
        "Contrast_level": contrast_val,
        "SNR1": snr1,
        "SNR2": snr2,
        "SNR3": snr3,
        "SNR4": snr4,
        "SNR5": snr5,
        "SNR6": snr6,
        "SNR7": snr7,
        "SNR8": snr8,
        "SNR9": snr9,
        "SNR10": snr10,
        "Foreground_mean": foreground_mean,
        "Background_mean": background_mean,
        "Foreground_var": foreground_var,
        "Background_var": background_var,
        "Mean_intensity": mean_intensity,
        "Std_intensity": std_intensity,
        "Variance_intensity": variance_intensity,
        "Michelson_contrast": michelson_contrast,
        "RMS_contrast": rms_contrast,
        "SSIM": ssim_score,
        "PSNR": psnr,
        "Edge_density": edge_density,
        "MI": mi,
        "NMI": nmi,
        "CE": ce
    }


''' 
Function to compute the entropy of an image
'''
def compute_image_entropy(img):
    # Load the image
    #img = io.imread(image_path)

    # Convert to grayscale if the image is colored
    if len(img.shape) > 2:
        img = np.mean(img, axis=2).astype(np.uint8)

    # Calculate the histogram of pixel values
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256), density=True)

    # Remove zero probability values (since log(0) is undefined)
    hist = hist[hist > 0]

    # Compute entropy using scipy.stats.entropy
    # The function computes entropy as -sum(pk * log(pk))
    # base=2 means we get the result in bits
    entropy_value = scipy.stats.entropy(hist, base=2)

    return entropy_value


'''
the method for computing all metrics over two folders with intensity and mask images
Note: the mask images should have the same name as the intensity images.
'''
def metrics(input_intensity_path, input_mask_path, output_filepath, set_index):

    noise_list = []
    contrast_list = []

    # Get the list of files in the specified directory
    intensity_files = os.listdir(input_intensity_path)

    # Sort the files by name in reverse order
    sorted_intensity_files = sorted(intensity_files, reverse=False)

    image_paths = []
    min_contrast_val = sys.maxsize
    min_noise_val = sys.maxsize
    min_contrast_str = ''
    min_noise_str = ''
    for file_path in sorted_intensity_files:
        image_paths.append(file_path)
        # base name
        bn = os.path.basename(file_path)
        parts = str.split(bn, "_")
        for index in range(0,len(parts)):
            if str(parts[index]).startswith(str("contrast")):
                # this is to avoid errors when converting 007.tiff to integer
                substring = parts[index + 1][0:3]
                contrast_list.append(substring)
                val = int(substring)
                if min_contrast_val > val:
                    min_contrast_val = val
                    min_contrast_str = parts[index + 1]


            if str(parts[index]).startswith(str("noise")):
                noise_list.append(parts[index + 1])
                substring = parts[index + 1][0:3]
                val = int(substring)
                if min_noise_val > val:
                    min_noise_val = val
                    min_noise_str = parts[index + 1]

    # Get the list of files in the specified directory
    mask_files = os.listdir(input_mask_path)
    # Sort the files by name in reverse order
    sorted_mask_files = sorted(mask_files, reverse=False)

    mask_paths = []
    for file_path in sorted_mask_files:
        mask_paths.append(file_path)


    # it should be set1_cex_noise_007_contrast_001.tiff
    reference_image_filename = os.path.basename(sorted_intensity_files[0])
    idx = reference_image_filename.find("contrast")
    idx += len("contrast_")
    reference_image_filename.replace(reference_image_filename[idx:],min_contrast_str)
    idx = reference_image_filename.find("noise")
    idx += len("noise_")
    reference_image_filename.replace(reference_image_filename[idx:],min_noise_str)
    print("INFO: reference_image_filename=", reference_image_filename)

    reference_fullimage_filename = os.path.join(input_intensity_path, reference_image_filename)
    print("INFO: reference_fullimage_filename= ", reference_fullimage_filename)

    reference_image = io.imread(reference_fullimage_filename)

    results = []
    for index in range(0,len(sorted_intensity_files)):
        maskimage_filename = os.path.join(input_mask_path, sorted_mask_files[index])
        print("INFO: mask_filename= ", maskimage_filename)
        mask = io.imread(maskimage_filename)

        intensityimage_filename = os.path.join(input_intensity_path, sorted_intensity_files[index])
        print("INFO: imtensity_filename= ", intensityimage_filename)
        img = io.imread(intensityimage_filename)

        noise_val = noise_list[index]
        contrast_val = contrast_list[index]

        print("INFO: ", noise_val, contrast_val, " index=", index)
        results.append(
            #calculate_all_metrics(img, mask, reference_image, noise_val, contrast_val, intensityimage_filename, set_index)
            calculate_all_metrics(img, mask, reference_image, noise_val, contrast_val, sorted_intensity_files[index],
                                  set_index)
        )

    #print(noise_list, contrast_list)

    # save the results to a CSV file
    df = pd.DataFrame(results)
    if output_filepath.__contains__("xls"):
        df.to_excel(output_filepath, index=False)
    elif output_filepath.__contains__("csv"):
        df.to_csv(output_filepath, index=False)



'''
The main method
it should be called as:
python metrics.py --input_intensity_path <path to folder with intensity images> --input_mask_path <path to folder with mask images> --output_filepath <path to output csv file>
Example:
python metrics.py --input_intensity_path ./set1 --input_mask_path ./bin_mask_set1 --output_filepath ./results.csv

'''
def main():
    parser = argparse.ArgumentParser(description="calculate SEM image quality metrics.")
    parser.add_argument(
        "--input_intensity_path",type=str,required=True,
        help="Path to the intensity images in a folder."
    )
    parser.add_argument(
        "--input_mask_path",type=str,required=True,
        help="Path to the mask images in a folder."
    )
    parser.add_argument(
        "--output_filepath",type=str,required=True,
        help="path of a CSV file.",
        default=f"../image_quality_metrics.csv"
    )
    parser.add_argument(
        "--set_index",type=int,required=False,
        help="index of the simulated set.",
        default=1
    )

    args = parser.parse_args()
    if args.input_intensity_path is None:
        print('ERROR: missing input intensity dir')
        return

    if args.input_mask_path is None:
        print('ERROR: missing input mask dir')
        return

    input_intensity_folder = args.input_intensity_path
    input_mask_folder = args.input_mask_path
    output_filepath = args.output_filepath
    set_index = args.set_index

    # if not os.path.exists(output_folder):
    #     # create the output folder
    #     os.mkdir(output_folder)
    #     print("INFO: created output folder = ", output_folder)

    print('Arguments:')
    print('image_intensities= {}'.format(input_intensity_folder))
    print('image_masks= {}'.format(input_mask_folder))
    print('output filepath = {}'.format(output_filepath))
    print('input set index = {}'.format(set_index))

    metrics(input_intensity_folder, input_mask_folder, output_filepath, set_index)


# a list of arguments:
#
# --input_intensity_path
# C:\PeterB\Projects\TestData\SEM_simulated_noise_contrast\artimagen_sim_noise_contrast_corrected\set1
# --input_mask_path
# C:\PeterB\Projects\TestData\SEM_simulated_noise_contrast\artimagen_sim_noise_contrast_corrected\bin_mask_set1
# --output_filepath
# C:\PeterB\Projects\TestData\SEM_simulated_noise_contrast\artimagen_sim_noise_contrast_corrected\test\results.csv
#
# test configuration:
# --input_intensity_path
# C:\PeterB\Projects\TestData\SEM_simulated_noise_contrast\artimagen_sim_noise_contrast_corrected\test\images
# --input_mask_path
# C:\PeterB\Projects\TestData\SEM_simulated_noise_contrast\artimagen_sim_noise_contrast_corrected\test\masks
# --output_filepath
# C:\PeterB\Projects\TestData\SEM_simulated_noise_contrast\artimagen_sim_noise_contrast_corrected\test\results.csv

if __name__ == "__main__":
    print('Python %s on %s' % (sys.version, sys.platform))
    #sys.path.extend([WORKING_DIR_AND_PYTHON_PATHS])

    main()


