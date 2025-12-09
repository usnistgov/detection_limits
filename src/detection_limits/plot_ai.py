'''
NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
'''

# Author: Peter Bajcsy
# Date: 2025-04-01
# Description: This script plots image quality metrics from SEM images with varying noise and contrast levels to evaluate signal-to-noise ratio (SNR) and other image quality metrics. The script reads data from a CSV file containing various metrics and generates scatter plots showing the relationship between noise, contrast, and different quality measures.
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

'''
metrics computed and saved in a format
IMAGE-NAME,DICE-COEFFICIENT,TRUE POSITIVE,TRUE NEGATIVE,FALSE POSITIVE,FALSE NEGATIVE,FNR, FPR
Set_index,Noise_level,Contrast_level,SNR1,SNR2,SNR3,SNR4,SNR5,SNR6,SNR7,SNR8,SNR9, SNR10,Foreground_mean,Background_mean,Foreground_var,Background_var,
Mean_intensity,Std_intensity,Variance_intensity,Michelson_contrast,RMS_contrast,SSIM,PSNR,Edge_density,MI,NMI,CE

'''


# the metrics in wanted_header_csv are the ones that are used to generate plots
# IMAGE-NAME,DICE-COEFFICIENT,TRUE POSITIVE,TRUE NEGATIVE,FALSE POSITIVE,FALSE NEGATIVE,Set_index,Noise_level,Contrast_level,SNR1,SNR2,SNR3,SNR4,SNR5,SNR6,Foreground_mean,Background_mean,Foreground_var,Background_var,Mean_intensity,Std_intensity,Variance_intensity,Michelson_contrast,RMS_contrast,SSIM,PSNR,Edge_density,MI,NMI,CE
# wanted_header_csv=["IMAGE-NAME","DICE-COEFFICIENT","TRUE POSITIVE","TRUE NEGATIVE","FALSE POSITIVE","FALSE NEGATIVE",
#                    "Set_index","Noise_level","Contrast_level","SNR4","SNR5","Michelson_contrast","RMS_contrast","SSIM",
#                    "PSNR","Edge_density","MI","NMI","CE"]

wanted_header_csv = ["IMAGE-NAME", "DICE-COEFFICIENT", "TRUE POSITIVE", "TRUE NEGATIVE", "FALSE POSITIVE", "FALSE NEGATIVE",
                     "Set_index", "Noise_level", "Contrast_level", "SNR1", "SNR2", "SNR3", "SNR4", "SNR5", "SNR6", "SNR7", "SNR8", "SNR9", "SNR10",
                     "Foreground_mean", "Background_mean", "Foreground_var", "Background_var", "Mean_intensity", "Std_intensity",
                     "Variance_intensity", "Michelson_contrast", "RMS_contrast", "SSIM", "PSNR", "Edge_density", "MI", "NMI", "CE"]

ai_header_csv = ["DICE-COEFFICIENT", "TRUE POSITIVE", "TRUE NEGATIVE", "FALSE POSITIVE", "FALSE NEGATIVE"]


def plot_3d_metrics(noise_values: np.ndarray, contrast_values: np.ndarray, metric_values: np.ndarray,
                    metric_name: str, log_scale: bool, output_filepath: Union[str, Path]) -> None:
    fig = plt.figure()
    # Plot metric_values as a function of noise and contrast
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Noise')
    ax.set_ylabel('Contrast')

    if log_scale:
        # Avoid log(0) or log(negative)
        safe_metric_values = np.where(metric_values > 0, metric_values, 1e-10)
        log_metric_values = np.log(safe_metric_values)

        sc = ax.scatter(noise_values, contrast_values, log_metric_values, c=log_metric_values, cmap='plasma', alpha=0.5)
        ax.set_zlabel(f'ln({metric_name})')
        ax.set_title(f'ln({metric_name}) = f(Noise and Contrast)')
        plt.colorbar(sc, label=f'ln({metric_name})')
    else:
        sc = ax.scatter(noise_values, contrast_values, metric_values, c=metric_values, cmap='plasma', alpha=0.5)
        ax.set_zlabel(metric_name)
        ax.set_title(f'{metric_name} = f(Noise and Contrast)')
        plt.colorbar(sc, label=metric_name)

    output_path = Path(output_filepath) / f"{metric_name}.png"
    plt.savefig(output_path)
    plt.close()


def plot_3d_planes(noise_values: np.ndarray, contrast_values: np.ndarray, metric_values: np.ndarray, human_snr_threshold: float,
                   fp_rate: np.ndarray, log_scale: bool, metric_name: str, output_filepath: Union[str, Path]) -> None:
    """
    Plot SNR4 (estimated SNR from data) based on noise and contrast and add the Rose criterion threshold plane
    and the False Negative rate threshold plane.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Noise')
    ax.set_ylabel('Contrast')

    xx, yy = np.meshgrid(np.linspace(noise_values.min(), noise_values.max(), 100),
                         np.linspace(contrast_values.min(), contrast_values.max(), 100))

    min_SNR = np.min(metric_values)
    max_SNR = np.max(metric_values)

    min_snr_fp = min_SNR
    max_snr_fp = max_SNR
    min_thresh_on_fp_rate = 0.0
    max_thresh_on_fp_rate = 0.0

    if "Dice" in metric_name:
        min_dice = 0.8
        logger.info(f"min_dice_rate={min_dice}")
        min_thresh_on_fp_rate = np.round(min_dice*100)/100.0

        logger.info(f"thresh_on_dice: {min_thresh_on_fp_rate}")
        snr_dice_array = metric_values[np.isclose(fp_rate, min_thresh_on_fp_rate, atol=0.1)]
        if len(snr_dice_array) == 0:
            min_snr_fp = min_SNR
            logger.info(f"SNR thresh_on_dice = 0.8: replaced by min SNR: {min_SNR}")
        else:
            min_snr_fp = np.average(snr_dice_array)

        max_dice = np.max(fp_rate)
        logger.info(f"max_dice={max_dice}")
        max_thresh_on_fp_rate = np.round(max_dice*100)/100.0

        snr_dice_fixed_array = metric_values[np.isclose(fp_rate, max_thresh_on_fp_rate, atol=0.1)]
        if len(snr_dice_fixed_array) == 0:
            max_snr_fp = max_SNR
            logger.info(f"SNR thresh_on_dice = 0.95: replaced by max SNR: {max_snr_fp}")
        else:
            max_snr_fp = np.average(snr_dice_fixed_array)
            logger.info(f"SNR thresh_on_dice = 0.95: {max_snr_fp}")

    elif "FPR" in metric_name or "FNR" in metric_name:
        # False positive rate is calculated by dividing the number of False Positives (FP) by the total number of negative samples (FP + TN).
        # A higher FPR indicates a model is prone to more errors, specifically making more false alarms (incorrectly identifying negatives as positives)
        # fp_rate = fp_values / (fp_values + tn_values)
        max_fp_rate = 0.1  # np.max(fp_rate)
        logger.info(f"max_fp_rate={max_fp_rate}")
        max_thresh_on_fp_rate = np.round(max_fp_rate*100)/100.0

        logger.info(f"thresh_on_fp_rate: {max_thresh_on_fp_rate}")
        snr_fp_rate_array = metric_values[np.isclose(fp_rate, max_thresh_on_fp_rate, atol=0.01)]
        if len(snr_fp_rate_array) == 0:
            max_snr_fp = min_SNR
            logger.info(f"SNR thresh_on_fp_rate = 0.1: replaced by min SNR: {max_snr_fp}")
        else:
            max_snr_fp = np.average(snr_fp_rate_array)

        min_fp_rate = np.min(fp_rate)
        logger.info(f"min_fp_rate={min_fp_rate}")
        min_thresh_on_fp_rate = np.round(min_fp_rate*100)/100.0
        snr_fp_fixed_array = metric_values[np.isclose(fp_rate, min_thresh_on_fp_rate, atol=0.01)]
        if len(snr_fp_fixed_array) == 0:
            min_snr_fp = max_SNR
            logger.info(f"SNR thresh_on_fp_rate = 0.05: replaced by min SNR: {min_snr_fp}")
        else:
            min_snr_fp = np.average(snr_fp_fixed_array)
            logger.info(f"SNR thresh_on_fp_rate = 0.05: {min_snr_fp}")

    if "Dice" in metric_name:
        show_name = "Dice"
        end_index = len(metric_name) - len("Dice") - 1
        plot_name = metric_name[0:end_index]
    elif "FNR" in metric_name:
        show_name = "FNR"
        end_index = len(metric_name) - len("FNR") - 1
        plot_name = metric_name[0:end_index]
    else:
        show_name = "FPR"
        end_index = len(metric_name) - len("FPR") - 1
        plot_name = metric_name[0:end_index]

        # Add text label for minimum SNR
        # thresh_on_fp_rate, (1.0 - thresh_on_fp_rate),
        ax.text2D(0.01, 0.95, f'For Thresh {show_name} = {max_thresh_on_fp_rate}, {plot_name}: ln({min_snr_fp:.3f})={np.log(min_snr_fp):.2f}',
                  transform=ax.transAxes, color='blue')
        ax.text2D(0.01, 0.90, f'For Min {show_name} = {min_thresh_on_fp_rate}, {plot_name}: ln({max_snr_fp:.3f})={np.log(max_snr_fp):.2f}',
                  transform=ax.transAxes, color='green')

    if log_scale:
        safe_metric_values = np.where(metric_values > 0, metric_values, 1e-10)
        log_metric_values = np.log(safe_metric_values)
        sc = ax.scatter(noise_values, contrast_values, log_metric_values, c=log_metric_values, cmap='plasma',
                        alpha=0.5)
        ax.set_zlabel(f'ln({plot_name})')
        ax.set_title(f'ln({plot_name}) = f(Noise and Contrast)')
        plt.colorbar(sc, label=f'ln({plot_name})', pad=0.2)

        zz = np.ones_like(xx) * np.log(human_snr_threshold if human_snr_threshold > 0 else 1e-10)
        zz_fp_min = np.ones_like(xx) * np.log(min_snr_fp if min_snr_fp > 0 else 1e-10)
        zz_fp_max = np.ones_like(xx) * np.log(max_snr_fp if max_snr_fp > 0 else 1e-10)
    else:
        sc = ax.scatter(noise_values, contrast_values, metric_values, c=metric_values, cmap='plasma', alpha=0.5)
        ax.set_zlabel(plot_name)
        ax.set_title(f'{plot_name} = f(Noise and Contrast)')
        plt.colorbar(sc, label=plot_name, pad=0.2)
        zz = np.ones_like(xx) * human_snr_threshold
        zz_fp_min = np.ones_like(xx) * min_snr_fp
        zz_fp_max = np.ones_like(xx) * max_snr_fp

    # Adding the Rose criterion threshold plane
    ax.plot_surface(xx, yy, zz, color='r', alpha=0.3)

    if "Dice" in metric_name:
        # max Dice is good
        ax.plot_surface(xx, yy, zz_fp_min, color='blue', alpha=0.3)
    else:
        # min FNR or FPR is good
        ax.plot_surface(xx, yy, zz_fp_max, color='blue', alpha=0.3)

    output_path = Path(output_filepath) / f"{metric_name}.png"
    plt.savefig(output_path)
    plt.close()


def plot_snr_dice_comparison(merged_csv_path: str, human_snr_threshold: float, log_scale: bool, root_output_filepath: str) -> None:
    """
    This method computes the confusion matrix and plots the SNR with the Dice, FPR, and FNR coefficient comparison
    """
    var_index = {}
    try:
        header = pd.read_csv(merged_csv_path, nrows=0).columns.tolist()
    except FileNotFoundError:
        logger.error(f"File not found: {merged_csv_path}")
        raise

    logger.info(f"Read CSV header: {header}")
    for item in wanted_header_csv:
        if item not in header:
            logger.error(f"Error: {item} not found in the CSV header")
            continue
        var_index.update({item: header.index(item)})

    logger.debug(f"dictionary with names and indices var_index={var_index}")

    # load the data
    try:
        data = np.genfromtxt(merged_csv_path, delimiter=',', skip_header=1)
    except Exception as e:
        logger.error(f"Error reading CSV data: {e}")
        raise

    if data.shape[1] < 22:
        logger.error("Error: number of columns expected = 23")
        raise ValueError("Incorrect number of columns in CSV")

    noise_values = data[:, var_index["Noise_level"]]
    contrast_values = data[:, var_index["Contrast_level"]]

    # information about the AI model accuracy
    tn_values = data[:, var_index["TRUE NEGATIVE"]]
    fn_values = data[:, var_index["FALSE NEGATIVE"]]
    tp_values = data[:, var_index["TRUE POSITIVE"]]
    fp_values = data[:, var_index["FALSE POSITIVE"]]
    dice_values = data[:, var_index["DICE-COEFFICIENT"]]

    # derive ai model quality metrics
    # False positive rate is calculated by dividing the number of False Positives (FP) by the total number of negative samples (FP + TN).
    # A higher FPR indicates a model is prone to more errors, specifically making more false alarms (incorrectly identifying negatives as positives)
    with np.errstate(divide='ignore', invalid='ignore'):
        fp_rate = np.divide(fp_values, (fp_values + tn_values))
        fn_rate = np.divide(fn_values, (tp_values + fn_values))
        fp_rate = np.nan_to_num(fp_rate)
        fn_rate = np.nan_to_num(fn_rate)

    ai_accuracy_metrics = ["Dice", "FPR", "FNR"]

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
        # print("DEBUG: elem=", elem, " index=", var_index[elem])
        # metric_name = elem
        # if elem == "IMAGE-NAME" or elem == "Set_index" or elem == "Noise_level" or elem == "Contrast_level":
        #     continue
        logger.debug(f"elem={elem} index={var_index[elem]}")

        if "SNR" not in str(elem):
            continue

        metric_name = snr_mapping.get(elem, elem)

        output_filepath = Path(root_output_filepath) / "3d_relation"
        if not output_filepath.exists():
            output_filepath.mkdir(parents=True, exist_ok=True)
            logger.info(f"created output folder = {output_filepath}")

        snr_values = data[:, var_index[elem]]
        for elem_ai in ai_accuracy_metrics:
            plot_name = f"{metric_name}_{elem_ai}"
            if elem_ai == "Dice":
                plot_3d_planes(noise_values, contrast_values, snr_values, human_snr_threshold, dice_values, log_scale, plot_name, output_filepath)
            elif elem_ai == "FPR":
                plot_3d_planes(noise_values, contrast_values, snr_values, human_snr_threshold, fp_rate, log_scale, plot_name, output_filepath)
            elif elem_ai == "FNR":
                plot_3d_planes(noise_values, contrast_values, snr_values, human_snr_threshold, fn_rate, log_scale, plot_name, output_filepath)


def plot_ai_model(merged_csv_path: str, human_snr_threshold: float, log_scale: bool, root_output_filepath: str) -> None:
    var_index = {}
    try:
        header = pd.read_csv(merged_csv_path, nrows=0).columns.tolist()
    except FileNotFoundError:
        logger.error(f"File not found: {merged_csv_path}")
        raise

    logger.info(f"Read CSV header: {header}")
    for item in wanted_header_csv:
        if item not in header:
            logger.error(f"Error: {item} not found in the CSV header")
            continue
        var_index.update({item: header.index(item)})

    logger.debug(f"dictionary with names and indices var_index={var_index}")

    # load the data
    try:
        data = np.genfromtxt(merged_csv_path, delimiter=',', skip_header=1)
    except Exception as e:
        logger.error(f"Error reading CSV data: {e}")
        raise

    if data.shape[1] < 22:
        logger.error("Error: number of columns expected = 23")
        raise ValueError("Incorrect number of columns in CSV")

    set_values = data[:, var_index["Set_index"]]
    noise_values = data[:, var_index["Noise_level"]]
    contrast_values = data[:, var_index["Contrast_level"]]

    # information about the AI model accuracy
    tn_values = data[:, var_index["TRUE NEGATIVE"]]
    fn_values = data[:, var_index["FALSE NEGATIVE"]]
    tp_values = data[:, var_index["TRUE POSITIVE"]]
    fp_values = data[:, var_index["FALSE POSITIVE"]]
    dice_values = data[:, var_index["DICE-COEFFICIENT"]]

    # derive ai model quality metrics
    # False positive rate is calculated by dividing the number of False Positives (FP) by the total number of negative samples (FP + TN).
    # A higher FPR indicates a model is prone to more errors, specifically making more false alarms (incorrectly identifying negatives as positives)
    with np.errstate(divide='ignore', invalid='ignore'):
        fp_rate_values = np.divide(fp_values, (fp_values + tn_values))
        fn_rate_values = np.divide(fn_values, (tp_values + fn_values))
        fp_rate_values = np.nan_to_num(fp_rate_values)
        fn_rate_values = np.nan_to_num(fn_rate_values)

        sum_values = tp_values + tn_values + fp_values + fn_values
        tp_prob = np.nan_to_num(np.divide(tp_values, sum_values))
        tn_prob = np.nan_to_num(np.divide(tn_values, sum_values))
        fp_prob = np.nan_to_num(np.divide(fp_values, sum_values))
        fn_prob = np.nan_to_num(np.divide(fn_values, sum_values))

    ai_accuracy_metrics = ["Dice", "FPR", "FNR", "FALSE POSITIVE", "FALSE NEGATIVE", "TRUE POSITIVE", "TRUE NEGATIVE"]

    # these graphs are without the Rose threshold and the corresponding ai model threshold
    log_scale_orig = log_scale
    log_scale = False

    for elem in ai_accuracy_metrics:
        metric_name = elem
        output_filepath = Path(root_output_filepath) / "3d_ai_metrics"
        if not output_filepath.exists():
            output_filepath.mkdir(parents=True, exist_ok=True)
            logger.info(f"created output folder = {output_filepath}")

        values_to_plot = None
        if elem == "TRUE POSITIVE":
            values_to_plot = tp_prob
        elif elem == "TRUE NEGATIVE":
            values_to_plot = tn_prob
        elif elem == "FALSE NEGATIVE":
            values_to_plot = fn_prob
        elif elem == "FALSE POSITIVE":
            values_to_plot = fp_prob
        elif elem == "Dice":
            values_to_plot = dice_values
        elif elem == "FPR":
            values_to_plot = fp_rate_values
        elif elem == "FNR":
            values_to_plot = fn_rate_values

        if values_to_plot is not None:
            plot_3d_metrics(noise_values, contrast_values, values_to_plot, metric_name, log_scale, output_filepath)

    log_scale = log_scale_orig

    for elem in ai_accuracy_metrics:
        metric_name = elem
        output_filepath = Path(root_output_filepath) / "2d_ai_metrics"
        if not output_filepath.exists():
            output_filepath.mkdir(parents=True, exist_ok=True)
            logger.info(f"created output folder = {output_filepath}")

        values_to_plot = None
        if elem == "TRUE POSITIVE":
            values_to_plot = tp_prob
        elif elem == "TRUE NEGATIVE":
            values_to_plot = tn_prob
        elif elem == "FALSE NEGATIVE":
            values_to_plot = fn_prob
        elif elem == "FALSE POSITIVE":
            values_to_plot = fp_prob
        elif elem == "Dice":
            values_to_plot = dice_values
        elif elem == "FPR":
            values_to_plot = fp_rate_values
        elif elem == "FNR":
            values_to_plot = fn_rate_values

        if values_to_plot is not None:
            plot_2d_variable(noise_values, contrast_values, values_to_plot, metric_name, log_scale, output_filepath)


def plot_confusion_matrix(merged_csv_path: str, output_filepath: Union[str, Path]) -> None:
    var_index = {}
    try:
        header = pd.read_csv(merged_csv_path, nrows=0).columns.tolist()
    except FileNotFoundError:
        logger.error(f"File not found: {merged_csv_path}")
        raise

    logger.info(f"Read CSV header: {header}")
    for item in wanted_header_csv:
        if item not in header:
            logger.error(f"Error: {item} not found in the CSV header")
            raise ValueError(f"{item} not found in CSV header")
        var_index.update({item: header.index(item)})

    logger.debug(f"dictionary with names and indices var_index={var_index}")

    try:
        data = np.genfromtxt(merged_csv_path, delimiter=',', skip_header=1)
    except Exception as e:
        logger.error(f"Error reading CSV data: {e}")
        raise

    if data.shape[1] < 22:
        logger.error("Error: number of columns expected = 23")
        raise ValueError("Incorrect number of columns in CSV")

    tn_values = data[:, var_index["TRUE NEGATIVE"]]
    fn_values = data[:, var_index["FALSE NEGATIVE"]]
    tp_values = data[:, var_index["TRUE POSITIVE"]]
    fp_values = data[:, var_index["FALSE POSITIVE"]]

    num_classes = 2
    conf_matrix = np.zeros((num_classes, num_classes))

    for i in range(0, len(tn_values)):
        tp = tp_values[i]
        tn = tn_values[i]
        fp = fp_values[i]
        fn = fn_values[i]
        conf_matrix[1, 1] += tp
        conf_matrix[0, 0] += tn
        conf_matrix[0, 1] += fp
        conf_matrix[1, 0] += fn

    with np.errstate(divide='ignore', invalid='ignore'):
        confusion_matrix_percentage = np.nan_to_num(np.round(100.0 * conf_matrix / np.sum(conf_matrix)))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_percentage, display_labels=['Negative', 'Positive'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Aggregated Confusion Matrix (Percentage)')

    output_path = Path(output_filepath) / 'aggregated_confusion_matrix.png'
    plt.savefig(output_path)
    plt.close()


def plot_2d_variable(noise_values: np.ndarray, contrast_values: np.ndarray, metric_values: np.ndarray,
                     metric_name: str, log_scale: bool, output_filepath: Union[str, Path]) -> None:
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
    plt.savefig(os.path.join(output_filepath, f"{metric_name}.png"))
    plt.close()


def main():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description='Script to plot AI accuracy = f(SEM simulated contrast and noise)')
    parser.add_argument('--merged_csv_filepath',
                        default='merged_ai_data_quality.csv',
                        type=str,
                        help='filepath to the merged (AI model and data quality) CSV file')
    parser.add_argument('--output_filepath',
                        default='.',
                        type=str,
                        help='filepath where the outputs will be saved.')
    parser.add_argument('--log_scale',
                        action='store_true',
                        help='Use logarithmic scale for noise values')

    args = parser.parse_args()

    if args.merged_csv_filepath is None:
        logger.error('missing merged_csv_filepath')
        return

    if args.output_filepath is None:
        logger.error('missing output_filepath')
        return

    merged_csv_filepath = args.merged_csv_filepath
    output_filepath = args.output_filepath

    output_path = Path(output_filepath)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"created output folder = {output_filepath}")

    human_snr_threshold = 5
    log_scale = True

    try:
        plot_snr_dice_comparison(merged_csv_filepath, human_snr_threshold, log_scale, output_filepath)
        # plot_ai_model(merged_csv_filepath, human_snr_threshold, log_scale, output_filepath)
        # plot_confusion_matrix(merged_csv_filepath, output_filepath)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
