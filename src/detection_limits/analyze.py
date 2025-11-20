'''
NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
'''

# Author: Peter Bajcsy
# Date: 2025-04-01
# Description: This script plots image quality metrics from SEM images with varying noise and contrast levels to evaluate signal-to-noise ratio (SNR) and other image quality metrics. The script reads data from a CSV file containing various metrics and generates scatter plots showing the relationship between noise, contrast, and different quality measures.
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

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

wanted_header_csv=["IMAGE-NAME","DICE-COEFFICIENT","TRUE POSITIVE","TRUE NEGATIVE","FALSE POSITIVE","FALSE NEGATIVE",
                   "Set_index","Noise_level","Contrast_level","SNR1","SNR2", "SNR3","SNR4", "SNR5","SNR6","SNR7", "SNR8","SNR9", "SNR10",
                   "Foreground_mean","Background_mean","Foreground_var","Background_var","Mean_intensity","Std_intensity",
                   "Variance_intensity","Michelson_contrast","RMS_contrast","SSIM","PSNR","Edge_density","MI","NMI","CE"]

ai_header_csv=["DICE-COEFFICIENT","TRUE POSITIVE","TRUE NEGATIVE","FALSE POSITIVE","FALSE NEGATIVE"]

def dice_to_snr(noise_values,contrast_values, metric_values, elem_ai, elem_ai_values, log_scale, metric_name, output_filepath):
    ############################################################################3
    # Plot SNR4 (estimated SNR from data) based on noise and contrast and add the Rose criterion threshold plane
    # and the False Negative rate threshold plane

    if elem_ai != "Dice" and elem_ai != "FPR" and elem_ai != "FNR":
        print("INFO: processing only Dice, FPR and FNR: elem_ai=", elem_ai, "metric_name=", metric_name)
        return

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
    #print("INFO: min_dice=", min_dice)
    max_dice = np.max(elem_ai_values)
    #print("INFO: max_dice_=", max_dice)

    snr_dice = []
    snr_sdev_dice = []
    number_of_samples = 100
    delta_proximity =2*(max_dice-min_dice)/(number_of_samples)

    print("INFO: delta_proximity=",delta_proximity)
    for dice in np.linspace(min_dice, max_dice, number_of_samples):
        #print(f"{dice:.1f}")
        #print("INFO: dice: ", dice)
        snr_dice_array = metric_values[np.isclose(elem_ai_values, dice, atol=delta_proximity)]
        #print(snr_dice_array)
        if len(snr_dice_array) == 0:
            avg_val = min_SNR
            print("INFO: SNR thresh_on_dice: replaced by min SNR:", min_SNR)
            continue
        else:
            avg_val = np.average(snr_dice_array)
            stdev_val = np.std(snr_dice_array)

        snr_dice.append(avg_val)
        snr_sdev_dice.append(stdev_val)
        print("INFO: avg_val=", avg_val, " stdev_val=", stdev_val)

    # save snr_dice to a CSV file, where the CSV filename is the metric name
    # Create a DataFrame with the dice values and corresponding SNR values
    df = pd.DataFrame({
        elem_ai: np.linspace(min_dice, max_dice, number_of_samples),
        metric_name: snr_dice
    })

    csv_name = metric_name
    # Extract the metric name for the filename
    if metric_name.__contains__("SNR"):
        if metric_name == "SNR1":
            csv_name = "SNR_power_est"
        elif metric_name == "SNR2":
            csv_name = "SNR_RMSpower_est"
        elif metric_name == "SNR3":
            csv_name = "SNR_invCV2_est"
        elif metric_name == "SNR4":
            csv_name = "SNR_invCV_est"
        elif metric_name == "SNR5":
            csv_name = "SNR_power_est"
        elif metric_name == "SNR6":
            csv_name = "SNR_invCV2_param"
        elif metric_name == "SNR7":
            csv_name = "SNR_power_param"
        elif metric_name == "SNR8":
            csv_name = "SNR_RMSpower_param"
        elif metric_name == "SNR9":
            csv_name = "Cohend_est"
        elif metric_name == "SNR10":
            csv_name = "Cohend_param"

    # save the DataFrame to a CSV file
    csv_filename = f"{csv_name}_vs_{elem_ai}.csv"
    csv_path = os.path.join(output_filepath, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f"INFO: Saved {csv_name} vs {elem_ai} relationship to {csv_path}")

    # plot Dice vs snr_dice
    plt.figure()
    #lt.plot(np.linspace(min_dice, max_dice, number_of_samples), snr_dice)
    # Create error bar plot
    plt.errorbar(np.linspace(min_dice, max_dice, number_of_samples), snr_dice, yerr=snr_sdev_dice, fmt='o-', capsize=5, ecolor='black',
                 markerfacecolor='blue', markersize=8, label='Data with std dev')
    plt.xlabel(elem_ai)
    plt.ylabel(metric_name)
    plt.title(metric_name+'=f(AI model '+elem_ai+')')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    figure_filename = f"{csv_name}_vs_{elem_ai}.png"
    plt.savefig(os.path.join(output_filepath, figure_filename))
    plt.close()


def dice_to_snr_mapping(merged_csv_path, human_snr_threshold, log_scale, root_output_filepath):
    var_index ={}
    header = pd.read_csv(merged_csv_path, nrows=0).columns.tolist()
    print("Read CSV header:",header)
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
        exit()

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
    fp_rate_values = fp_values / (fp_values + tn_values)
    fn_rate_values = fn_values / (tp_values + fn_values)

    ai_accuracy_metrics = ["Dice", "FPR", "FNR"]
    for elem in var_index:
        print("DEBUG: elem=", elem, " index=", var_index[elem])
        metric_name = elem
        # if elem == "IMAGE-NAME" or elem == "Set_index" or elem == "Noise_level" or elem == "Contrast_level":
        #     continue

        if not str(elem).__contains__("SNR"):
            continue

        if elem == "SNR1":
            metric_name = "SNR_power_est"
        if elem == "SNR2":
            metric_name = "SNR_RMSpower_est"
        if elem == "SNR3":
            metric_name = "SNR_invCV2_est"
        if elem == "SNR4":
            metric_name = "SNR_invCV_est"
        if elem == "SNR5":
            metric_name = "SNR_invCV_param"
        if elem == "SNR6":
            metric_name = "SNR_invCV2_param"
        if elem == "SNR7":
            metric_name = "SNR_power_param"
        if elem == "SNR8":
            metric_name = "SNR_RMSpower_param"
        if elem == "SNR9":
            metric_name = "Cohend_est"
        if elem == "SNR10":
            metric_name = "Cohend_param"


        output_filepath = os.path.join(root_output_filepath, "dice2snr_mapping")
        if not os.path.exists(output_filepath):
            os.mkdir(output_filepath)
            print("INFO: created output folder = ", output_filepath)

        snr_values = data[:, var_index[elem]]
        for elem_ai in ai_accuracy_metrics:
            if elem_ai == "Dice":
                dice_to_snr(noise_values, contrast_values, snr_values, elem_ai, dice_values, log_scale, metric_name,
                               output_filepath)
            if elem_ai == "FPR":
                dice_to_snr(noise_values, contrast_values, snr_values, elem_ai, fp_rate_values, log_scale, metric_name,
                               output_filepath)
            if elem_ai == "FNR":
                dice_to_snr(noise_values, contrast_values, snr_values, elem_ai, fn_rate_values, log_scale,
                               metric_name, output_filepath)


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


    parser.add_argument('--log_scale',
                    action='store_true',
                    help='Use logarithmic scale for noise values')

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
        # create the output folder
        os.mkdir(output_filepath)
        print("INFO: created output folder = ", output_filepath)

    num_classes = 2
    #plot_confusion_matrix(csv_path, output_dirpath, num_classes)
    human_snr_threshold = 5
    log_scale = True
    dice_to_snr_mapping(merged_csv_filepath, human_snr_threshold, log_scale, output_filepath)



if __name__ == '__main__':
    main()

