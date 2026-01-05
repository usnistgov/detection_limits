'''
NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
'''

# Author: Peter Bajcsy and Pushkar Sathe
# Date: 2025-04-01
# Description: This script computes image quality metrics from SEM images

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import scipy.stats
from skimage import feature, io
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

'''
compute image SNR quality metrics based on the intensity and mask pairs of images
'''


def calculate_all_snr_with_mask(image: np.ndarray, mask: np.ndarray, noise: Optional[float] = None) -> \
        Tuple[float, float, float, float, float, float, float, float, float, float, float, float, float, float]:
    """
    Compute image SNR quality metrics based on the intensity and mask pairs of images.

    Args:
        image (np.ndarray): The input image.
        mask (np.ndarray): The mask image.
        noise (Optional[float]): The noise level.

    Returns:
        Tuple[float, ...]: A tuple containing various SNR metrics and statistics.
    """
    # Normalize the image first
    # normalized_image = normalize_image(image)
    normalized_image = image.copy()
    # Use boolean indexing directly
    foreground_region = normalized_image[mask > 0]
    background_region = normalized_image[mask == 0]

    # Handle empty regions to avoid RuntimeWarnings
    def get_stats(arr):
        return (np.mean(arr), np.var(arr), np.std(arr)) if arr.size > 0 else (0.0, 0.0, 0.0)

    foreground_mean, foreground_var, foreground_std = get_stats(foreground_region)
    background_mean, background_var, background_std = get_stats(background_region)

    # Helper for safe division
    def safe_div(n, d):
        return n / d if d != 0 and not np.isnan(d) else 0.0

    noise_val = noise if noise is not None else 0.0
    noise_sq = noise_val * noise_val

    # SNR calculations
    snr1 = safe_div(foreground_var, background_var)
    snr2 = safe_div(np.sqrt(foreground_mean), np.sqrt(foreground_var))
    snr3 = safe_div(foreground_mean**2, background_std**2)
    snr4 = safe_div(foreground_mean, background_std)
    snr5 = safe_div(foreground_mean, noise_val)
    snr6 = safe_div(foreground_mean**2, noise_sq)
    snr7 = safe_div(foreground_var, noise_sq)
    snr8 = safe_div(np.sqrt(foreground_mean), noise_val)
    snr9 = safe_div(foreground_mean - background_mean, background_std)
    snr10 = safe_div(foreground_mean - background_mean, noise_val)

    return (
        float(snr1), float(snr2), float(snr3), float(snr4), float(snr5),
        float(snr6), float(snr7), float(snr8), float(snr9), float(snr10),
        float(foreground_mean), float(background_mean),
        float(foreground_var), float(background_var)
    )


def calculate_all_metrics(image: np.ndarray, mask: Optional[np.ndarray] = None, reference_image: Optional[np.ndarray] = None,
                          noise_val: Optional[Union[float, str]] = None, contrast_val: Optional[Union[float, str]] = None,
                          intensityimage_filename: str = "test.tif", set_index: int = 1) -> dict:
    """
    Calculates various image quality metrics from loaded image, mask, and reference intensity image.

    Args:
        image (np.ndarray): The input image.
        mask (Optional[np.ndarray]): The mask image.
        reference_image (Optional[np.ndarray]): The reference intensity image.
        noise_val (Optional[Union[float, str]]): The noise value.
        contrast_val (Optional[Union[float, str]]): The contrast value.
        intensityimage_filename (str): The filename of the intensity image.
        set_index (int): The index of the set.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    image_flat = image.flatten()

    # Single image metrics
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    variance_intensity = np.var(image)

    min_val = float(np.min(image))
    max_val = float(np.max(image))

    if np.abs(max_val + min_val) > 0.0001:
        michelson_contrast = (max_val - min_val) / (max_val + min_val)
    else:
        michelson_contrast = 0.0

    rms_contrast = std_intensity

    # Edge density metric
    edges = feature.canny(image)
    edge_density = np.sum(edges) / image.size
    logger.debug(f"np.sum(edges)={np.sum(edges)}")

    # Pair-wise image metrics with reference intensity image
    if reference_image is not None:
        data_range = image.max() - image.min()
        if data_range == 0:
            data_range = 1.0  # Avoid error in ssim if image is constant
        ssim_score = ssim(image, reference_image, data_range=data_range)
        psnr = cv2.PSNR(image, reference_image)
    else:
        smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)
        data_range = image.max() - image.min()
        if data_range == 0:
            data_range = 1.0
        ssim_score = ssim(image, smoothed_image, data_range=data_range)
        psnr = cv2.PSNR(image, smoothed_image)

    # Pair-wise image metrics with reference segmentation mask image
    mi = None
    nmi = None
    ce = None

    if mask is not None:
        mask_flat = mask.flatten()
        mi = mutual_info_score(image_flat, mask_flat)
        nmi = normalized_mutual_info_score(image_flat, mask_flat)
        ce = compute_image_entropy(image)

    try:
        noise_float = float(noise_val) if noise_val is not None and str(noise_val) != "N/A" else 0.0
    except ValueError:
        noise_float = 0.0

    (snr1, snr2, snr3, snr4, snr5, snr6, snr7, snr8, snr9, snr10, foreground_mean, background_mean,
     foreground_var, background_var) = calculate_all_snr_with_mask(image, mask, noise=noise_float)

    intensityimage_ome_filename = str(Path(intensityimage_filename).with_suffix("")) + ".ome.tif"

    return {
        "IMAGE-NAME": intensityimage_ome_filename,
        "Set_index": set_index,
        "Noise_level": noise_val,
        "Contrast_level": contrast_val,
        "SNR1": snr1, "SNR2": snr2, "SNR3": snr3, "SNR4": snr4, "SNR5": snr5,
        "SNR6": snr6, "SNR7": snr7, "SNR8": snr8, "SNR9": snr9, "SNR10": snr10,
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


def compute_image_entropy(img: np.ndarray) -> float:
    """
    Compute the entropy of an image.

    Args:
        img (np.ndarray): The input image.

    Returns:
        float: The entropy value measured in bits.
    """
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

    return float(entropy_value)


def metrics(input_intensity_source: Union[str, List[str]], input_mask_source: Union[str, List[str]], output_filepath: str, set_index: int) -> None:
    """
    Compute all metrics over two folders with intensity and mask images.

    Note:
        The mask images should have the same name as the intensity images.

    Args:
        input_intensity_source (Union[str, List[str]]): Directory path or list of intensity image paths.
        input_mask_source (Union[str, List[str]]): Directory path or list of mask image paths.
        output_filepath (str): Destination file path for the metrics CSV/XLS output.
        set_index (int): Index of the simulated set.
    """
    # Handle input_intensity_source
    if isinstance(input_intensity_source, str) and Path(input_intensity_source).is_dir():
        sorted_intensity_files = sorted([str(p) for p in Path(input_intensity_source).iterdir() if p.is_file()])
    elif isinstance(input_intensity_source, list):
        sorted_intensity_files = sorted(input_intensity_source)
    else:
        raise ValueError("input_intensity_source must be a directory path or a list of file paths")

    # Handle input_mask_source
    if isinstance(input_mask_source, str) and Path(input_mask_source).is_dir():
        sorted_mask_files = sorted([str(p) for p in Path(input_mask_source).iterdir() if p.is_file()])
    elif isinstance(input_mask_source, list):
        sorted_mask_files = sorted(input_mask_source)
    else:
        raise ValueError("input_mask_source must be a directory path or a list of file paths")

    file_info = []
    for f in sorted_intensity_files:
        bn = Path(f).name
        parts = bn.split("_")
        c_val = sys.maxsize
        n_val = sys.maxsize
        c_str = ""
        n_str = ""

        for i, part in enumerate(parts):
            if part == "contrast" and i + 1 < len(parts):
                try:
                    c_str = parts[i+1]
                    c_val = int(c_str[:3])
                except ValueError:
                    pass
            if part == "noise" and i + 1 < len(parts):
                try:
                    n_str = parts[i+1]
                    n_val = int(n_str[:3])
                except ValueError:
                    pass

        file_info.append({
            "path": f,
            "contrast_val": c_val,
            "noise_val": n_val,
            "contrast_str": c_str,
            "noise_str": n_str
        })

    if not file_info:
        logger.error("No files to process.")
        return

    min_c = min(item["contrast_val"] for item in file_info)
    min_n = min(item["noise_val"] for item in file_info)

    # Find reference image: the one with min contrast and min noise
    ref_image_path = None
    for item in file_info:
        if item["contrast_val"] == min_c and item["noise_val"] == min_n:
            ref_image_path = item["path"]
            break

    if ref_image_path is None:
        logger.warning(f"Reference image with min contrast {min_c} and min noise {min_n} not found. Using first image.")
        ref_image_path = sorted_intensity_files[0]

    logger.info(f"reference_fullimage_filename= {ref_image_path}")
    reference_image = io.imread(ref_image_path)

    results = []
    for i, item in enumerate(file_info):
        mask_path = sorted_mask_files[i] if i < len(sorted_mask_files) else None
        if mask_path is None:
            logger.warning(f"No mask found for index {i}")
            continue

        logger.info(f"mask_filename= {mask_path}")
        mask = io.imread(mask_path)

        intensity_path = item["path"]
        logger.info(f"intensity_filename= {intensity_path}")
        img = io.imread(intensity_path)

        noise_val = item["noise_val"] if item["noise_val"] != sys.maxsize else "N/A"
        contrast_val = item["contrast_val"] if item["contrast_val"] != sys.maxsize else "N/A"

        logger.info(f"{noise_val} {contrast_val} index={i}")

        results.append(calculate_all_metrics(img, mask, reference_image, noise_val, contrast_val, intensity_path, set_index))

    df = pd.DataFrame(results)
    if output_filepath.endswith(".xls") or output_filepath.endswith(".xlsx"):
        df.to_excel(output_filepath, index=False)
    else:
        df.to_csv(output_filepath, index=False)


def main():
    parser = argparse.ArgumentParser(description="Calculate SEM image quality metrics.")
    parser.add_argument("--input_intensity_path", type=str, required=True, help="Path to the intensity images folder.")
    parser.add_argument("--input_mask_path", type=str, required=True, help="Path to the mask images folder.")
    parser.add_argument("--output_filepath", type=str, required=True, default="../image_quality_metrics.csv", help="Path of the output CSV file.")
    parser.add_argument("--set_index", type=int, required=False, default=1, help="Index of the simulated set.")

    args = parser.parse_args()

    input_intensity_folder = args.input_intensity_path
    input_mask_folder = args.input_mask_path
    output_filepath = args.output_filepath
    set_index = args.set_index

    logger.info('Arguments:')
    logger.info(f'image_intensities= {input_intensity_folder}')
    logger.info(f'image_masks= {input_mask_folder}')
    logger.info(f'output filepath = {output_filepath}')
    logger.info(f'input set index = {set_index}')

    try:
        metrics(input_intensity_folder, input_mask_folder, output_filepath, set_index)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger.info(f'Python {sys.version} on {sys.platform}')
    main()
