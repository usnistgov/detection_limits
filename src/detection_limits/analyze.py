'''
NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
'''

# Authors: Peter Bajcsy, Pushkar Sathe
# Created: 2025-04-01
# Description: This script plots image quality metrics from SEM images
# with varying noise and contrast levels to evaluate signal-to-noise ratio
# (SNR) and other image quality metrics. The script reads data from a CSV
# file containing various metrics and generates scatter plots showing the
# relationship between noise, contrast, and different quality measures.
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from .report import Report

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


def compute_dice_to_snr_mapping(metric_values, elem_ai, elem_ai_values, metric_name):
    ############################################################################
    # Plot SNR4 (estimated SNR from data) based on noise and contrast and add the Rose criterion threshold plane
    # and the False Negative rate threshold plane

    if elem_ai != "Dice" and elem_ai != "FPR" and elem_ai != "FNR":
        print("INFO: processing only Dice, FPR and FNR: elem_ai=", elem_ai, "metric_name=", metric_name)
        return None

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel('Noise')
    # ax.set_ylabel('Contrast')

    # xx, yy = np.meshgrid(np.linspace(noise_values.min(), noise_values.max(), 100),
    #                      np.linspace(contrast_values.min(), contrast_values.max(), 100))

    # this value is the default when max Dice, min FPR, or min FNR cannot be matched with SNR metrics
    min_SNR = np.min(metric_values)
    max_SNR = np.max(metric_values)
    min_dice = np.min(elem_ai_values)
    # print("INFO: min_dice=", min_dice)
    max_dice = np.max(elem_ai_values)
    # print("INFO: max_dice_=", max_dice)

    snr_dice = []
    snr_sdev_dice = []
    number_of_samples = 100
    delta_proximity = 2 * (max_dice - min_dice) / (number_of_samples)

    print("INFO: delta_proximity=", delta_proximity)
    for dice in np.linspace(min_dice, max_dice, number_of_samples):
        # print(f"{dice:.1f}")
        # print("INFO: dice: ", dice)
        snr_dice_array = metric_values[np.isclose(elem_ai_values, dice, atol=delta_proximity)]
        # print(snr_dice_array)
        if len(snr_dice_array) == 0:
            avg_val = min_SNR
            print("INFO: SNR thresh_on_dice: replaced by min SNR:", min_SNR)
            stdev_val = 0.0
        else:
            avg_val = np.average(snr_dice_array)
            stdev_val = np.std(snr_dice_array)

        snr_dice.append(avg_val)
        snr_sdev_dice.append(stdev_val)
        print("INFO: avg_val=", avg_val, " stdev_val=", stdev_val)

    # save snr_dice to a CSV file, where the CSV filename is the metric name
    # Create a DataFrame with the dice values, corresponding SNR values, and std dev
    df = pd.DataFrame({
        elem_ai: np.linspace(min_dice, max_dice, number_of_samples),
        metric_name: snr_dice,
        f"{metric_name}_std": snr_sdev_dice
    })
    return df


def generate_report(merged_csv_path, human_snr_threshold=5.0, log_scale=True) -> Report:
    """Generate an in-memory report from a merged CSV file.

    This is the canonical analysis entry point for building report artifacts.
    """
    if log_scale and human_snr_threshold <= 0:
        raise ValueError("human_snr_threshold must be > 0 when log_scale is enabled")

    report = Report()
    report.set_plot_options(human_snr_threshold=human_snr_threshold, log_scale=log_scale)
    var_index = {}
    header = pd.read_csv(merged_csv_path, nrows=0).columns.tolist()
    print("Read CSV header:", header)
    for item in wanted_header_csv:
        if item not in header:
            print(f"Error: {item} not found in the CSV header")
            continue
        var_index.update({item: header.index(item)})

    print("DEBUG: dictionary with names and indices var_index=", var_index)

    # get the indices of ai model entries defined in ai_header_csv
    ai_var_index = {}
    for item in ai_header_csv:
        if item not in header:
            continue
        ai_var_index.update({item: header.index(item)})

    # load the data
    data = np.genfromtxt(merged_csv_path, delimiter=',', skip_header=1)
    if data.shape[1] < 22:
        print("Error: number of columns expected = 23")
        # In a library function, we should probably raise an exception instead of exit()
        raise ValueError("Number of columns in CSV is less than expected 23")

    # information about the AI model accuracy
    tn_values = data[:, var_index["TRUE NEGATIVE"]]
    fn_values = data[:, var_index["FALSE NEGATIVE"]]
    tp_values = data[:, var_index["TRUE POSITIVE"]]
    fp_values = data[:, var_index["FALSE POSITIVE"]]
    dice_values = data[:, var_index["DICE-COEFFICIENT"]]

    # derive ai model quality metrics
    # False positive rate is calculated by dividing the number of False Positives (FP) by the total number of negative samples (FP + TN).
    # A higher FPR indicates a model is prone to more errors, specifically making more false alarms (incorrectly identifying negatives as positives)
    fp_rate_values = fp_values / (fp_values + tn_values)
    fn_rate_values = fn_values / (tp_values + fn_values)

    ai_accuracy_metrics = {"Dice": dice_values,
                           "FPR": fp_rate_values,
                           "FNR": fn_rate_values}

    for elem in var_index:
        print("DEBUG: elem=", elem, " index=", var_index[elem])
        metric_name = elem
        # if elem == "IMAGE-NAME" or elem == "Set_index" or elem == "Noise_level" or elem == "Contrast_level":
        #     continue

        if not str(elem).__contains__("SNR"):
            continue

        if elem == "SNR1":
            metric_name = "SNR_power_est"
        elif elem == "SNR2":
            metric_name = "SNR_RMSpower_est"
        elif elem == "SNR3":
            metric_name = "SNR_invCV2_est"
        elif elem == "SNR4":
            metric_name = "SNR_invCV_est"
        elif elem == "SNR5":
            metric_name = "SNR_invCV_param"
        elif elem == "SNR6":
            metric_name = "SNR_invCV2_param"
        elif elem == "SNR7":
            metric_name = "SNR_power_param"
        elif elem == "SNR8":
            metric_name = "SNR_RMSpower_param"
        elif elem == "SNR9":
            metric_name = "Cohend_est"
        elif elem == "SNR10":
            metric_name = "Cohend_param"

        snr_values = data[:, var_index[elem]]
        for elem_ai, ai_values in ai_accuracy_metrics.items():
            df = compute_dice_to_snr_mapping(snr_values, elem_ai, ai_values, metric_name)
            if df is not None:
                report.add_mapping(metric_name, elem_ai, df)

    return report


def main():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description='Script to create a mapping from AI accuracy to SNR ')
    parser.add_argument('--merged_csv_filepath',
                        default='merged_ai_data_quality.csv',
                        type=str,
                        help='filepath to the merged (AI model and data quality) CSV file')
    parser.add_argument('--output_filepath',
                        default='.',
                        type=str,
                        help='filepath where the outputs will be saved.')
    parser.add_argument('--human_snr_threshold',
                        default=5.0,
                        type=float,
                        help='SNR threshold used as a visual reference in report plots (default: 5.0)')
    parser.add_argument('--log_scale',
                        dest='log_scale',
                        action='store_true',
                        default=True,
                        help='Plot SNR values on a logarithmic scale (default: enabled)')
    parser.add_argument('--no_log_scale',
                        dest='log_scale',
                        action='store_false',
                        help='Disable logarithmic scaling in report plots')
    parser.add_argument('--save_csvs',
                        action='store_true',
                        help='Save mapping CSV files in addition to figures')

    args = parser.parse_args()

    if args.merged_csv_filepath is None:
        print('ERROR: missing merged_csv_filepath')
        return

    if args.output_filepath is None:
        print('ERROR: missing output_filepath')
        return

    merged_csv_filepath = args.merged_csv_filepath
    output_filepath = args.output_filepath
    if not os.path.exists(output_filepath):
        os.mkdir(output_filepath)
        print("INFO: created output folder = ", output_filepath)

    report = generate_report(merged_csv_filepath,
                             human_snr_threshold=args.human_snr_threshold,
                             log_scale=args.log_scale)
    report.save(output_filepath, save_csvs=args.save_csvs)


if __name__ == '__main__':
    main()
