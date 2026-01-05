'''
NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
'''

# Author: Peter Bajcsy
# Date: 2025-04-01
# Description: This script generates 2D plots of image quality metrics from SEM images with varying noise and contrast levels
# to evaluate signal-to-noise ratio (SNR) and other image quality metrics.
# The script reads data from a CSV file containing various metrics and generates scatter plots showing the relationship
# between noise, contrast, and different quality measures.
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

'''
metrics computed and saved in a format
    IMAGE-NAME,Set_index,Noise_level,Contrast_level,SNR1,SNR2,SNR3,SNR4,SNR5,SNR6,SNR7,SNR8,SNR9, SNR10,
    Foreground_mean,Background_mean,Foreground_var,Background_var,Mean_intensity,Std_intensity,Variance_intensity,
    Michelson_contrast,RMS_contrast,SSIM,PSNR,Edge_density,MI,NMI,CE
 
'''

# wanted_header_csv=["IMAGE-NAME","Set_index","Noise_level","Contrast_level","SNR1","SNR2", "SNR3","SNR4", "SNR5","SNR6",
#                    "SNR7", "SNR8","SNR9", "SNR10",
#                    "Foreground_mean","Background_mean","Foreground_var","Background_var","Mean_intensity","Std_intensity",
#                    "Variance_intensity","Michelson_contrast","RMS_contrast","SSIM","PSNR","Edge_density","MI","NMI","CE"]

wanted_header_csv = ["IMAGE-NAME", "DICE-COEFFICIENT", "TRUE POSITIVE", "TRUE NEGATIVE", "FALSE POSITIVE", "FALSE NEGATIVE",
                     "Set_index", "Noise_level", "Contrast_level", "SNR1", "SNR2", "SNR3", "SNR4", "SNR5", "SNR6", "SNR7", "SNR8", "SNR9", "SNR10",
                     "Foreground_mean", "Background_mean", "Foreground_var", "Background_var", "Mean_intensity", "Std_intensity",
                     "Variance_intensity", "Michelson_contrast", "RMS_contrast", "SSIM", "PSNR", "Edge_density", "MI", "NMI", "CE"]

# ai_header_csv = ["IMAGE-NAME",	"DICE-COEFFICIENT",	"TRUE POSITIVE",	"TRUE NEGATIVE",	"FALSE POSITIVE",	"FALSE NEGATIVE"]


def plot_2d_variable(noise_values: np.ndarray, contrast_values: np.ndarray, metric_values: np.ndarray,
                     log_scale: bool, metric_name: str, output_filepath: Union[str, Path]) -> None:
    """
    Generates a 2D scatter plot of a specific metric against noise and contrast.

    Args:
        noise_values (np.ndarray): Array of noise levels.
        contrast_values (np.ndarray): Array of contrast levels.
        metric_values (np.ndarray): Array of metric values to plot as color.
        log_scale (bool): Whether to use a logarithmic scale for the noise axis.
        metric_name (str): Name of the metric being plotted.
        output_filepath (Union[str, Path]): Directory path to save the output plot.
    """
    plt.figure(figsize=(10, 6))
    if log_scale:
        # Avoid log(0)
        safe_noise = np.where(noise_values > 0, noise_values, 1e-10)
        plt.scatter(np.log(safe_noise), contrast_values, c=metric_values, cmap='plasma', alpha=0.5)
        plt.xlabel('ln(Noise)')
    else:
        plt.scatter(noise_values, contrast_values, c=metric_values, cmap='plasma', alpha=0.5)
        plt.xlabel('Noise')

    plt.ylabel('Contrast')
    plt.colorbar(label=metric_name)
    plt.title(metric_name)
    plt.grid(True)
    output_path = Path(output_filepath) / f"{metric_name}.png"
    plt.savefig(output_path)
    plt.close()


def plot_2d_data_quality(csv_path: str, root_output_filepath: str, log_scale: bool) -> None:
    """Plot results from a CSV file

    Args:
        csv_path (str): Path to the CSV file containing the data.
        root_output_filepath (str): Root directory where the output plots will be saved.
        log_scale (bool): Boolean indicating whether to use a logarithmic scale for the noise axis.

    Raises:
        ValueError: If the CSV file does not contain the expected number of columns.
    """
    var_index = {}
    try:
        header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    except FileNotFoundError:
        logger.error(f"File not found: {csv_path}")
        raise
    logger.info(f"Read CSV header: {header}")
    for item in wanted_header_csv:
        if item not in header:
            logger.warning(f"Warning: {item} not found in the CSV header")
            continue
        var_index.update({item: header.index(item)})

    logger.debug(f"dictionary with names and indices var_index={var_index}")

    # load the data
    try:
        data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    except Exception as e:
        logger.error(f"Error reading CSV data: {e}")
        raise

    if data.shape[1] < 22:
        logger.error("Error: number of columns expected = 23")
        raise ValueError("Incorrect number of columns in CSV")

    noise_values = data[:, var_index["Noise_level"]]
    contrast_values = data[:, var_index["Contrast_level"]]

    ##################################

    snr_mapping = {
        "SNR1": "SNR_power_est",
        "SNR2": "SNR_RMSpower_est",
        "SNR3": "SNR_invCV2_est",
        "SNR4": "SNR_invCV_est",
        "SNR5": "SNR_invCV_param",
        "SNR6": "SNR_invCV2_param",
        "SNR7": "SNR_power_param",
        "SNR8": "SNR_RMSpower_param",
        "SNR9": "Cohend_est",
        "SNR10": "Cohend_param"
    }

    for elem in var_index:
        logger.debug(f"elem={elem} index={var_index[elem]}")
        metric_name = elem
        if elem in ["IMAGE-NAME", "Set_index", "Noise_level", "Contrast_level"]:
            continue

        metric_name = snr_mapping.get(elem, elem)

        output_filepath = Path(root_output_filepath) / "2d_dataquality_graphs"
        if not output_filepath.exists():
            output_filepath.mkdir(parents=True, exist_ok=True)
            logger.info(f"created output folder = {output_filepath}")

        plot_2d_variable(noise_values, contrast_values, data[:, var_index[elem]], log_scale, metric_name, output_filepath)


def plot_3d_variable(noise_values: np.ndarray, contrast_values: np.ndarray, metric_values: np.ndarray, metric_name: str,
                     log_scale: bool, output_filepath: Union[str, Path]) -> None:
    """
    Generates a 3D scatter plot of a specific metric against noise and contrast.

    Args:
        noise_values (np.ndarray): Array of noise levels.
        contrast_values (np.ndarray): Array of contrast levels.
        metric_values (np.ndarray): Array of metric values to plot on the z-axis.
        metric_name (str): Name of the metric being plotted.
        log_scale (bool): Whether to use a logarithmic scale for the metric values.
        output_filepath (Union[str, Path]): Directory path to save the output plot.
    """
    fig = plt.figure()
    # Plot metric_values as a function of noise and contrast
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Noise')
    ax.set_ylabel('Contrast')

    if log_scale:
        # Avoid log(0)
        safe_metric_values = np.where(metric_values > 0, metric_values, 1e-10)
        log_metric_values = np.log(safe_metric_values)

        sc = ax.scatter(noise_values, contrast_values, log_metric_values, c=log_metric_values, cmap='plasma',
                        alpha=0.5)
        ax.set_zlabel(f'ln({metric_name})')
        ax.set_title(f'ln({metric_name}) = f(Noise and Contrast)')
        plt.colorbar(sc, label=f'ln({metric_name})')
    else:
        sc = ax.scatter(noise_values, contrast_values, metric_values, c=metric_values, cmap='plasma', alpha=0.5)
        ax.set_zlabel(metric_name)
        ax.set_title(f'{metric_name} = f(Noise and Contrast)')
        plt.colorbar(sc, label=metric_name)

    # xx, yy = np.meshgrid(np.linspace(noise_values.min(), noise_values.max(), 100),
    #                      np.linspace(contrast_values.min(), contrast_values.max(), 100))

    # Adding the AI Hallucination criterion threshold plane
    # zz = np.ones_like(xx) * 0.05
    # ax.plot_surface(xx, yy, zz, color='gray', alpha=0.3)

    plt.savefig(os.path.join(output_filepath, (metric_name+'.png')))
    output_path = Path(output_filepath) / f"{metric_name}.png"
    plt.savefig(output_path)
    plt.close()


def plot_3d_data_quality(csv_path: str, root_output_filepath: str, log_scale: bool) -> None:
    """
    Plots all image data quality metrics in 3D.

    Args:
        csv_path (str): Path to the CSV file containing the data.
        root_output_filepath (str): Root directory where the output plots will be saved.
        log_scale (bool): Whether to use a logarithmic scale for the metric values.
    """
    var_index = {}
    try:
        header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    except FileNotFoundError:
        logger.error(f"File not found: {csv_path}")
        raise

    logger.info(f"Read CSV header: {header}")
    for item in wanted_header_csv:
        if item not in header:
            logger.warning(f"Warning: {item} not found in the CSV header")
            continue
        var_index.update({item: header.index(item)})

    logger.debug(f"dictionary with names and indices var_index={var_index}")

    # load the data
    try:
        data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    except Exception as e:
        logger.error(f"Error reading CSV data: {e}")
        raise

    if data.shape[1] < 22:
        logger.error("Error: number of columns expected = 23")
        raise ValueError("Incorrect number of columns in CSV")

    noise_values = data[:, var_index["Noise_level"]]
    contrast_values = data[:, var_index["Contrast_level"]]

    snr_mapping = {
        "SNR1": "SNR_power_est",
        "SNR2": "SNR_RMSpower_est",
        "SNR3": "SNR_invCV2_est",
        "SNR4": "SNR_invCV_est",
        "SNR5": "SNR_invCV_param",
        "SNR6": "SNR_invCV2_param",
        "SNR7": "SNR_power_param",
        "SNR8": "SNR_RMSpower_param",
        "SNR9": "Cohend_est",
        "SNR10": "Cohend_param"
    }
    #################################################################################
    # these graphs are without the Rose threshold and the corresponding ai model threshold
    for elem in var_index:
        logger.debug(f"elem={elem} index={var_index[elem]}")
        metric_name = elem
        if elem in ["IMAGE-NAME", "Set_index", "Noise_level", "Contrast_level"]:
            continue

        metric_name = snr_mapping.get(elem, elem)

        output_filepath = Path(root_output_filepath) / "3d_dataquality_graphs"
        if not output_filepath.exists():
            output_filepath.mkdir(parents=True, exist_ok=True)
            logger.info(f"created output folder = {output_filepath}")

        plot_3d_variable(noise_values, contrast_values, data[:, var_index[elem]], metric_name, log_scale, output_filepath)


def plot_snr_vs_metrics(csv_path: str, output_dirpath: Union[str, Path], log_scale: bool = True) -> None:
    """
    Plot relationships between SNR and other image quality metrics
    This method is for pair-wise comparison - could be insightful

    Args:
        csv_path: Path to the CSV file with metrics data
        output_dirpath: Directory to save output plots
        log_scale: Whether to use logarithmic scale for noise values
    """
    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        raise

    # Extract metrics for comparison
    metrics_to_compare = ["SNR4", "SNR5", "Michelson_contrast", "SSIM", "Edge_density", "MI"]

    # Filter metrics that actually exist in the data
    available_metrics = [m for m in metrics_to_compare if m in data.columns]

    if len(available_metrics) < 2:
        logger.warning("Not enough metrics found in CSV for comparison plots.")
        return

    # Create scatter plots comparing SNR with other metrics
    for i, metric1 in enumerate(available_metrics):
        for j, metric2 in enumerate(available_metrics[i+1:], i+1):
            plt.figure(figsize=(10, 6))
            plt.scatter(data[metric1], data[metric2], c=data["Noise_level"], cmap='viridis', alpha=0.7)
            plt.colorbar(label='Noise Level')
            plt.xlabel(metric1)
            plt.ylabel(metric2)
            plt.title(f'Relationship between {metric1} and {metric2}')
            plt.grid(True)

            output_path = Path(output_dirpath) / f'{metric1}_vs_{metric2}.png'
            plt.savefig(output_path)
            plt.close()

    # Plot SNR metrics against noise at different contrast levels
    if "Contrast_level" in data.columns:
        contrast_levels = sorted(data["Contrast_level"].unique())
        for metric in ["SNR4", "SNR5"]:
            if metric not in data.columns:
                continue

            plt.figure(figsize=(12, 8))
            for contrast in contrast_levels:
                subset = data[data["Contrast_level"] == contrast]
                x_values = np.log(subset["Noise_level"]) if log_scale else subset["Noise_level"]
                plt.plot(x_values, subset[metric], 'o-', label=f'Contrast={contrast}')

            plt.legend()
            plt.xlabel('ln(Noise)' if log_scale else 'Noise')
            plt.ylabel(metric)
            plt.title(f'{metric} vs Noise at different Contrast Levels')
            plt.grid(True)

            output_path = Path(output_dirpath) / f'{metric}_vs_noise_by_contrast.png'
            plt.savefig(output_path)
            plt.close()


def main():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description='Script to plot SNR = f(SEM simulated contrast and noise)')
    parser.add_argument('--csv_filepath',
                        default='snr_propertieslist.csv',
                        type=str,
                        help='filepath to the CSV file')
    parser.add_argument('--output_dirpath',
                        default='results',
                        type=str,
                        help='filepath to where the outputs will be saved.')
    parser.add_argument(
        "--set_index", type=int, required=False,
        help="index of the simulated set.",
        default=1
    )

    parser.add_argument('--log_scale',
                        action='store_true',
                        help='Use logarithmic scale for noise values')

    args = parser.parse_args()
    csv_path = args.csv_filepath
    root_dirpath = args.output_dirpath

    if not os.path.exists(root_dirpath):
        # create the output folder
        os.mkdir(root_dirpath)
        logger.info(f"created root output folder = {root_dirpath}")

    try:
        plot_2d_data_quality(csv_path, root_dirpath, log_scale=True)
        plot_3d_data_quality(csv_path, root_dirpath, log_scale=True)

        # this method is for pair-wise comparison
        output_dirpath = Path(root_dirpath) / "2d_dataquality_compgraphs"
        if not output_dirpath.exists():
            output_dirpath.mkdir(parents=True, exist_ok=True)
            logger.info(f"created output folder = {output_dirpath}")

        plot_snr_vs_metrics(csv_path, output_dirpath, log_scale=True)

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
