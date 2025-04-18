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
IMAGE-NAME,DICE-COEFFICIENT,TRUE POSITIVE,TRUE NEGATIVE,FALSE POSITIVE,FALSE NEGATIVE,
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

def plot_3d_metrics(noise_values,contrast_values, metric_values, metric_name, log_scale, output_filepath):

    fig = plt.figure()
    # Plot metric_values as a function of noise and contrast
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Noise')
    ax.set_ylabel('Contrast')

    if log_scale:
        sc = ax.scatter(noise_values, contrast_values, np.log(metric_values), c=np.log(metric_values), cmap='plasma',
                        alpha=0.5)
        ax.set_zlabel('ln('+metric_name+')')
        ax.set_title('ln('+metric_name+') = f(Noise and Contrast)')
        plt.colorbar(sc, label='ln('+metric_name+')')
    else:
        sc = ax.scatter(noise_values, contrast_values, metric_values, c=metric_values, cmap='plasma', alpha=0.5)
        ax.set_zlabel(metric_name)
        ax.set_title(metric_name + ' = f(Noise and Contrast)')
        plt.colorbar(sc, label=metric_name)


    # xx, yy = np.meshgrid(np.linspace(noise_values.min(), noise_values.max(), 100),
    #                      np.linspace(contrast_values.min(), contrast_values.max(), 100))

    # Adding the AI Hallucination criterion threshold plane
    # zz = np.ones_like(xx) * 0.05
    # ax.plot_surface(xx, yy, zz, color='gray', alpha=0.3)

    plt.savefig(os.path.join(output_filepath, (metric_name+'_vs_Noise_Contrast.png')))
    plt.close()


# The false negative rate (FNR), also known as the miss rate,
# represents the proportion of actual positive cases that are incorrectly classified as negative by a model.
# FNR = FN / (TP + FN)

# False positive rate is calculated by dividing the number of False Positives (FP) by the total number of negative samples (FP + TN).
# A higher FPR indicates a model is prone to more errors, specifically making more false alarms (incorrectly identifying negatives as positives)
def plot_3d_planes(noise_values,contrast_values, metric_values, human_snr_threshold, fp_rate, log_scale, metric_name, output_filepath):
    ############################################################################3
    # Plot SNR4 (estimated SNR from data) based on noise and contrast and add the Rose criterion threshold plane
    # and the False Negative rate threshold plane
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Noise')
    ax.set_ylabel('Contrast')

    xx, yy = np.meshgrid(np.linspace(noise_values.min(), noise_values.max(), 100),
                         np.linspace(contrast_values.min(), contrast_values.max(), 100))

    # Calculate and add the AI Hallucination criterion plane
    # total_pixels = tn_values + tp_values + fn_values + fp_values

    # False positive rate is calculated by dividing the number of False Positives (FP) by the total number of negative samples (FP + TN).
    # A higher FPR indicates a model is prone to more errors, specifically making more false alarms (incorrectly identifying negatives as positives)
    #fp_rate = fp_values / (fp_values + tn_values)
    min_fp_rate = np.min(fp_rate)
    print("INFO: min_fp_rate=", min_fp_rate)

    temp = int(min_fp_rate*100.0)
    thresh_on_fp_rate = np.mod(temp, 5)
    thresh_on_fp_rate = (temp + (5- thresh_on_fp_rate))/100.0

    print("INFO: thresh_on_fp_rate: ", thresh_on_fp_rate)
    min_snr_fp = metric_values[np.isclose(fp_rate, thresh_on_fp_rate, atol=0.005)].min()


    if metric_name.__contains__("Dice"):
        show_name = "Dice"
        end_index = len(metric_name) - len("Dice") - 1
        plot_name = metric_name[0:end_index]
    elif metric_name.__contains__("FNR"):
        show_name = "FNR"
        end_index = len(metric_name) - len("FNR") - 1
        plot_name = metric_name[0:end_index]
    else:
        show_name = "FPR"
        end_index = len(metric_name) - len("FPR") - 1
        plot_name = metric_name[0:end_index]

    # Add text label for minimum SNR
    # thresh_on_fp_rate, (1.0 - thresh_on_fp_rate),
    ax.text2D(0.05, 0.95,
              f'Min {plot_name} for {show_name} <= {thresh_on_fp_rate}: {min_snr_fp:.2f}', transform=ax.transAxes, color='blue')

    if log_scale:
        sc = ax.scatter(noise_values, contrast_values, np.log(metric_values), c=np.log(metric_values), cmap='plasma',
                        alpha=0.5)
        ax.set_zlabel('ln('+plot_name+')')
        ax.set_title('ln('+plot_name+') = f(Noise and Contrast)')
        plt.colorbar(sc, label='ln('+plot_name+')')
        zz = np.ones_like(xx) * np.log(human_snr_threshold)
        zz_fp = np.ones_like(xx) * np.log(min_snr_fp)
    else:
        sc = ax.scatter(noise_values, contrast_values, metric_values, c=metric_values, cmap='plasma', alpha=0.5)
        ax.set_zlabel(plot_name)
        ax.set_title(plot_name +' = f(Noise and Contrast)')
        plt.colorbar(sc, label=plot_name)
        zz = np.ones_like(xx) * human_snr_threshold
        zz_fp = np.ones_like(xx) * min_snr_fp

    # Adding the Rose criterion threshold plane
    ax.plot_surface(xx, yy, zz, color='r', alpha=0.3)

    # Adding the AI Hallucination criterion threshold plane
    ax.plot_surface(xx, yy, zz_fp, color='blue', alpha=0.3)

    plt.savefig(os.path.join(output_filepath, (metric_name+'_vs_Noise_Contrast.png')))
    plt.close()



'''
This method computes the confusion matrix and plots the SNR and Dice coefficient comparison
'''
def plot_snr_dice_comparison(merged_csv_path, human_snr_threshold, log_scale, output_filepath):
    var_index ={}
    header = pd.read_csv(merged_csv_path, nrows=0).columns.tolist()
    print("Read CSV header:",header)
    for item in wanted_header_csv:
        if item not in header:
            print(f"Error: {item} not found in the CSV header")
            exit()
        var_index.update({item: header.index(item)})

    print("DEBUG: dictionary with names and indices var_index=", var_index)

    # load the data
    data = np.genfromtxt(merged_csv_path, delimiter=',', skip_header=1)
    if data.shape[1] < 22:
        print("Error: number of columns expected = 23")
        exit()

    set_values = data[:, var_index["Set_index"]]
    noise_values = data[:, var_index["Noise_level"]]
    contrast_values = data[:, var_index["Contrast_level"]]

    # information about the data quality metrics
    # snr1_values = data[:, var_index["SNR1"]]
    # snr2_values = data[:, var_index["SNR2"]]
    # snr3_values = data[:, var_index["SNR3"]]
    snr4_values = data[:, var_index["SNR4"]]
    snr5_values = data[:, var_index["SNR5"]]
    # snr6_values = data[:, var_index["SNR6"]]
    michelson_values = data[:,var_index["Michelson_contrast"]]
    rms_values = data[:, var_index["RMS_contrast"]]
    ssim_values = data[:, var_index["SSIM"]]
    edge_values = data[:, var_index["Edge_density"]]
    mi_values = data[:, var_index["MI"]]

    # information about the AI model accuracy
    tn_values = data[:, var_index["TRUE NEGATIVE"]]
    fn_values = data[:, var_index["FALSE NEGATIVE"]]
    tp_values = data[:, var_index["TRUE POSITIVE"]]
    fp_values = data[:, var_index["FALSE POSITIVE"]]
    dice_values = data[:, var_index["DICE-COEFFICIENT"]]

    # derive ai model quality metrics
    # False positive rate is calculated by dividing the number of False Positives (FP) by the total number of negative samples (FP + TN).
    # A higher FPR indicates a model is prone to more errors, specifically making more false alarms (incorrectly identifying negatives as positives)
    fp_rate = fp_values / (fp_values + tn_values)
    fn_rate = fn_values / (tp_values + fn_values)

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

        snr_values = data[:, var_index[elem]]
        for elem_ai in ai_accuracy_metrics:
            if elem_ai == "Dice":
                plot_name = metric_name + '_Dice'
                plot_3d_planes(noise_values, contrast_values, snr_values, human_snr_threshold, dice_values, log_scale, plot_name,
                               output_filepath)
            if elem_ai == "FPR":
                plot_name = metric_name + '_FPR'
                plot_3d_planes(noise_values, contrast_values, snr_values, human_snr_threshold, fp_rate, log_scale, plot_name,
                               output_filepath)
            if elem_ai == "FNR":
                plot_name = metric_name + '_FNR'
                plot_3d_planes(noise_values, contrast_values, snr_values, human_snr_threshold, fn_rate, log_scale,
                               plot_name,
                               output_filepath)

    # ############################################################################
    # # Plot SNR4 (estimated SNR from data) based on noise and contrast and add the Rose criterion threshold plane
    # # and the False Positive rate threshold plane
    # fp_rate = fp_values / (fp_values + tn_values)
    # metric_name = 'SNR_est_FPR'
    # plot_3d_planes(noise_values, contrast_values, snr4_values, human_snr_threshold, fp_rate, log_scale, metric_name,
    #                output_filepath)
    # ############################################################################
    # # Plot SNR4 (estimated SNR from data) based on noise and contrast and add the Rose criterion threshold plane
    # # and the False Negative rate threshold plane
    # fn_rate = fn_values / (tp_values + fn_values)
    # metric_name = 'SNR_invCV_est_FNR'
    # plot_3d_planes(noise_values, contrast_values, snr4_values, human_snr_threshold, fn_rate, log_scale, metric_name,
    #                output_filepath)
    # ############################################################################
    # # Plot SNR4 (estimated SNR from data) based on noise and contrast and add the Rose criterion threshold plane
    # # and the Dice threshold plane
    # # fn_rate = fn_values / (tp_values + fn_values)
    # metric_name = 'SNR_est_Dice'
    # plot_3d_planes(noise_values, contrast_values, snr4_values, human_snr_threshold, dice_values, log_scale, metric_name,
    #                output_filepath)
    #
    # ############################################################################
    # # Plot SNR5 ( SNR from sim noise) as a function of noise and contrast and add the Rose criterion threshold plane
    # # and the False Positive rate threshold plane
    # fp_rate = fp_values / (fp_values + tn_values)
    # metric_name = 'SNR_param_FPR'
    # plot_3d_planes(noise_values, contrast_values, snr5_values, human_snr_threshold, fp_rate, log_scale, metric_name,
    #                output_filepath)
    # ############################################################################
    # # Plot SNR5 ( SNR from sim noise) as a function of noise and contrast and add the Rose criterion threshold plane
    # # and the False Negative rate threshold plane
    # fn_rate = fn_values / (tp_values + fn_values)
    # metric_name = 'SNR_param_FNR'
    # plot_3d_planes(noise_values, contrast_values, snr5_values, human_snr_threshold, fn_rate, log_scale, metric_name,
    #                output_filepath)
    # ############################################################################
    # # Plot Mutual Information  as a function of noise and contrast and add the Rose criterion threshold plane
    # # and the False Positive rate threshold plane
    # fp_rate = fp_values / (fp_values + tn_values)
    # metric_name = 'SNR_param_FPR'
    # plot_3d_planes(noise_values, contrast_values, mi_values, human_snr_threshold, fp_rate, log_scale, metric_name,
    #                output_filepath)
    # ############################################################################
    # # Plot SNR5 ( SNR from sim noise) as a function of noise and contrast and add the Rose criterion threshold plane
    # # and the Dice threshold plane
    # # fn_rate = fn_values / (tp_values + fn_values)
    # metric_name = 'SNR_param_Dice'
    # plot_3d_planes(noise_values, contrast_values, snr5_values, human_snr_threshold, dice_values, log_scale, metric_name,
    #                output_filepath)
    #
    ############################################################################
    # # Plot Mutual Information as a function of noise and contrast and add the Rose criterion threshold plane
    # # and the False Negative rate threshold plane
    # fn_rate = fn_values / (tp_values + fn_values)
    # metric_name = 'MI_FNR'
    # plot_3d_planes(noise_values, contrast_values, mi_values, human_snr_threshold, fn_rate, log_scale, metric_name,
    #                output_filepath)

    #################################################################################
    # these graphs are without the Rose threshold and the corresponding ai model threshold
    for elem in var_index:
        print("DEBUG: elem=", elem, " index=", var_index[elem])
        metric_name = elem
        if elem == "IMAGE-NAME" or elem == "Set_index" or elem == "Noise_level" or elem == "Contrast_level":
            continue

        if str(elem).__contains__("SNR"):
            continue
        #
        # if elem == "SNR1":
        #     metric_name = "SNR_power_est"
        # if elem == "SNR2":
        #     metric_name = "SNR_RMSpower_est"
        # if elem == "SNR3":
        #     metric_name = "SNR_invCV2_est"
        # if elem == "SNR4":
        #     metric_name = "SNR_invCV_est"
        # if elem == "SNR5":
        #     metric_name = "SNR_invCV_param"
        # if elem == "SNR6":
        #     metric_name = "SNR_invCV2_param"
        # if elem == "SNR7":
        #     metric_name = "SNR_power_param"
        # if elem == "SNR8":
        #     metric_name = "SNR_RMSpower_param"
        # if elem == "SNR9":
        #     metric_name = "Cohend_est"
        # if elem == "SNR10":
        #     metric_name = "Cohend_param"

        plot_3d_metrics(noise_values, contrast_values, data[:,var_index[elem]], metric_name, log_scale, output_filepath)


    ##################################################################
    # # Plot Dice as a function of noise and contrast
    # metric_name = 'Dice_coefficient'
    # plot_3d_metrics(noise_values, contrast_values, dice_values, metric_name, log_scale, output_filepath)
    # ##################################################################
    # # Plot Mutual Information as a function of noise and contrast
    # metric_name = 'MI'
    # plot_3d_metrics(noise_values, contrast_values, mi_values, metric_name, log_scale, output_filepath)
    # ##################################################################
    # # Plot Michelson contrast as a function of noise and contrast
    # metric_name = 'Michelson_contrast'
    # plot_3d_metrics(noise_values, contrast_values, michelson_values, metric_name, log_scale, output_filepath)
    #
    # ##################################################################
    # # Plot Mutual Information as a function of noise and contrast
    # metric_name = 'Edge_density'
    # plot_3d_metrics(noise_values, contrast_values, edge_values, metric_name, log_scale, output_filepath)
    # # Plot SSIM as a function of noise and contrast
    # metric_name = 'SSIM'
    # plot_3d_metrics(noise_values, contrast_values, ssim_values, metric_name, log_scale, output_filepath)
    # # Plot RMS contrast as a function of noise and contrast
    # metric_name = 'RMS_contrast'
    # plot_3d_metrics(noise_values, contrast_values, rms_values, metric_name, log_scale, output_filepath)
    # # Plot False Negative Rate as a function of noise and contrast
    # metric_name = 'FN_Rate'
    # fn_rate = fn_values / (tp_values + fn_values)
    # plot_3d_metrics(noise_values, contrast_values, fn_rate, metric_name, log_scale, output_filepath)
    # # Plot False Positive Rate as a function of noise and contrast
    # metric_name = 'FP_Rate'
    # plot_3d_metrics(noise_values, contrast_values, fp_rate, metric_name, log_scale, output_filepath)
    #

    ####################################################
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # sc = ax.scatter(noise_values, contrast_values, fp_rate, c=fp_rate, cmap='plasma', alpha=0.5)
    # plt.colorbar(sc, label='FP Rate')
    # ax.set_xlabel('Noise')
    # ax.set_ylabel('Contrast')
    # ax.set_zlabel('FP Rate')
    # ax.set_title('FP Rate = f(Noise and Contrast)')
    #
    # xx, yy = np.meshgrid(np.linspace(noise_values.min(), noise_values.max(), 100),
    #                      np.linspace(contrast_values.min(), contrast_values.max(), 100))
    #
    # # Adding the AI Hallucination criterion threshold plane
    # zz = np.ones_like(xx) * 0.05
    # ax.plot_surface(xx, yy, zz, color='gray', alpha=0.3)
    #
    # plt.savefig(os.path.join(output_filepath, 'FP_vs_Noise_Contrast.png'))
    # plt.close()
    #
    # ####################################
    # # Plot Dice as a function of noise and contrast
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # sc = ax.scatter(noise_values, contrast_values, dice_values, c=dice_values, cmap='plasma', alpha=0.5)
    # plt.colorbar(sc, label='AI Model Dice')
    # ax.set_xlabel('Noise')
    # ax.set_ylabel('Contrast')
    # ax.set_zlabel('Dice Coefficient')
    # ax.set_title('Dice = f(Noise and Contrast)')
    #
    # # Adding the AI Hallucination criterion threshold plane
    # zz = np.ones_like(xx) * 0.05
    # ax.plot_surface(xx, yy, zz, color='gray', alpha=0.3)
    #
    # plt.savefig(os.path.join(output_filepath, 'Dice_vs_Noise_Contrast.png'))
    # plt.close()
    #


'''
This method is for generating the aggregated confusion matrix
'''
def plot_confusion_matrix(merged_csv_path, output_filepath):
    # plot confusion matrix
    var_index = {}
    header = pd.read_csv(merged_csv_path, nrows=0).columns.tolist()
    print("Read CSV header:", header)
    for item in wanted_header_csv:
        if item not in header:
            print(f"Error: {item} not found in the CSV header")
            exit()
        var_index.update({item: header.index(item)})

    print("DEBUG: dictionary with names and indices var_index=", var_index)

    # load the data
    data = np.genfromtxt(merged_csv_path, delimiter=',', skip_header=1)
    if data.shape[1] < 22:
        print("Error: number of columns expected = 23")
        exit()

    # information about the AI model accuracy
    tn_values = data[:, var_index["TRUE NEGATIVE"]]
    fn_values = data[:, var_index["FALSE NEGATIVE"]]
    tp_values = data[:, var_index["TRUE POSITIVE"]]
    fp_values = data[:, var_index["FALSE POSITIVE"]]

    num_classes = 2
    conf_matrix = np.zeros((num_classes, num_classes))

    for i in range(0,len(tn_values)):
        tp = tp_values[i]
        tn = tn_values[i]
        fp = fp_values[i]
        fn = fn_values[i]
        conf_matrix[1, 1] += tp
        conf_matrix[0, 0] += tn
        conf_matrix[0, 1] += fp
        conf_matrix[1, 0] += fn
    confusion_matrix_percentage = np.round(100.0 * conf_matrix / np.sum(conf_matrix))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_percentage, display_labels=['Negative', 'Positive'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Aggregated Confusion Matrix (Percentage)')
    plt.savefig(os.path.join(output_filepath, 'aggregated_confusion_matrix.png'))
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
    plot_snr_dice_comparison(merged_csv_filepath, human_snr_threshold, log_scale, output_filepath)
    plot_confusion_matrix(merged_csv_filepath, output_filepath)

    # --input_ai_model
    # C:\PeterB\Projects\TestData\SEM_simulated_noise_contrast\artimagen_sim_noise_contrast_corrected\Merged_DataQuality_Accuracy\model7\dice-coef-per-image.csv
    # --input_data_quality
    # C:\PeterB\Projects\TestData\SEM_simulated_noise_contrast\artimagen_sim_noise_contrast_corrected\Merged_DataQuality_Accuracy\model7\data_quality_results_set6.csv
    # --output_filepath
    # C:\PeterB\Projects\TestData\SEM_simulated_noise_contrast\artimagen_sim_noise_contrast_corrected\Merged_DataQuality_Accuracy\model7\merged_ai_data_quality.csv



if __name__ == '__main__':
    main()

