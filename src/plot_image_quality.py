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
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
# from sklearn.metrics import ConfusionMatrixDisplay

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

wanted_header_csv=["IMAGE-NAME","DICE-COEFFICIENT","TRUE POSITIVE","TRUE NEGATIVE","FALSE POSITIVE","FALSE NEGATIVE",
                   "Set_index","Noise_level","Contrast_level","SNR1","SNR2", "SNR3","SNR4", "SNR5","SNR6","SNR7", "SNR8","SNR9", "SNR10",
                   "Foreground_mean","Background_mean","Foreground_var","Background_var","Mean_intensity","Std_intensity",
                   "Variance_intensity","Michelson_contrast","RMS_contrast","SSIM","PSNR","Edge_density","MI","NMI","CE"]

# ai_header_csv = ["IMAGE-NAME",	"DICE-COEFFICIENT",	"TRUE POSITIVE",	"TRUE NEGATIVE",	"FALSE POSITIVE",	"FALSE NEGATIVE"]

def plot_2d_variable(noise_values,contrast_values,metric_values,log_scale, metric_name, output_filepath):
    ##################################
    plt.figure(figsize=(10, 6))
    if log_scale:
        plt.scatter(np.log(noise_values), contrast_values, c=metric_values, cmap='plasma', alpha=0.5)
        plt.xlabel('ln(Noise)')
    else:
        plt.scatter(noise_values, contrast_values, c=metric_values, cmap='plasma', alpha=0.5)
        plt.xlabel('Noise')

    plt.ylabel('Contrast')
    plt.colorbar(label=metric_name)
    plt.title(metric_name)
    plt.grid(True)
    plt.savefig(os.path.join(output_filepath, metric_name+'.png'))
    plt.close()

######################################################################
# The method for plotting results from a CSV file
def plot_2d_data_quality(csv_path, root_output_filepath, log_scale):
    var_index ={}
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    print("Read CSV header:",header)
    for item in wanted_header_csv:
        if item not in header:
            print(f"Warning: {item} not found in the CSV header")
            continue
        var_index.update({item: header.index(item)})

    print("DEBUG: dictionary with names and indices var_index=", var_index)

    # load the data
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    if data.shape[1] < 22:
        print("Error: number of columns expected = 23")
        exit()


    noise_values = data[:, var_index["Noise_level"]]
    contrast_values = data[:, var_index["Contrast_level"]]
    # set_values = data[:, var_index["Set_index"]]

    ##################################
    for elem in var_index:
        print("DEBUG: elem=", elem, " index=", var_index[elem])
        metric_name = elem
        if elem == "IMAGE-NAME" or elem == "Set_index" or elem == "Noise_level" or elem == "Contrast_level":
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

        output_filepath = os.path.join(root_output_filepath, "2d_dataquality_graphs")
        if not os.path.exists(output_filepath):
            os.mkdir(output_filepath)
            print("INFO: created output folder = ", output_filepath)

        plot_2d_variable(noise_values, contrast_values, data[:,var_index[elem]], log_scale, metric_name, output_filepath)


def plot_3d_variable(noise_values,contrast_values, metric_values, metric_name, log_scale, output_filepath):

    fig = plt.figure()
    # Plot metric_values as a function of noise and contrast
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Noise')
    ax.set_ylabel('Contrast')

    if log_scale:
        log_metric_values = np.log(metric_values) if np.all(metric_values > 0) else np.log(metric_values + 1e-10)
        sc = ax.scatter(noise_values, contrast_values, log_metric_values, c=log_metric_values, cmap='plasma',
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

    plt.savefig(os.path.join(output_filepath, (metric_name+'.png')))
    plt.close()



'''
This method plots in 3D all image data quality metrics 
csv_path, output_dirpath, log_scale
'''
def plot_3d_data_quality(csv_path, root_output_filepath, log_scale):
    var_index ={}
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    print("Read CSV header:",header)
    for item in wanted_header_csv:
        if item not in header:
            print(f"Warning: {item} not found in the CSV header")
            continue
        var_index.update({item: header.index(item)})

    print("DEBUG: dictionary with names and indices var_index=", var_index)

    # load the data
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    if data.shape[1] < 22:
        print("Error: number of columns expected = 23")
        exit()

    set_values = data[:, var_index["Set_index"]]
    noise_values = data[:, var_index["Noise_level"]]
    contrast_values = data[:, var_index["Contrast_level"]]

    # # information about the data quality metrics
    # # snr1_values = data[:, var_index["SNR1"]]
    # # snr2_values = data[:, var_index["SNR2"]]
    # # snr3_values = data[:, var_index["SNR3"]]
    # snr4_values = data[:, var_index["SNR4"]]
    # snr5_values = data[:, var_index["SNR5"]]
    # # snr6_values = data[:, var_index["SNR6"]]
    # michelson_values = data[:,var_index["Michelson_contrast"]]
    # rms_values = data[:, var_index["RMS_contrast"]]
    # ssim_values = data[:, var_index["SSIM"]]
    # edge_values = data[:, var_index["Edge_density"]]
    # mi_values = data[:, var_index["MI"]]


    #################################################################################
    # these graphs are without the Rose threshold and the corresponding ai model threshold
    for elem in var_index:
        print("DEBUG: elem=", elem, " index=", var_index[elem])
        metric_name = elem
        if elem == "IMAGE-NAME" or elem == "Set_index" or elem == "Noise_level" or elem == "Contrast_level":
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


        output_filepath = os.path.join(root_output_filepath, "3d_dataquality_graphs")
        if not os.path.exists(output_filepath):
            os.mkdir(output_filepath)
            print("INFO: created output folder = ", output_filepath)

        plot_3d_variable(noise_values, contrast_values, data[:,var_index[elem]], metric_name, log_scale, output_filepath)



'''
this method is for pair-wise comparison - could be insightful
'''
def plot_snr_vs_metrics(csv_path, output_dirpath, log_scale=True):
    """
    Plot relationships between SNR and other image quality metrics

    Args:
        csv_path: Path to the CSV file with metrics data
        output_dirpath: Directory to save output plots
        log_scale: Whether to use logarithmic scale for noise values
    """
    data = pd.read_csv(csv_path)

    # Extract metrics for comparison
    metrics_to_compare = ["SNR4", "SNR5", "Michelson_contrast", "SSIM", "Edge_density", "MI"]

    # Create scatter plots comparing SNR with other metrics
    for i, metric1 in enumerate(metrics_to_compare):
        for j, metric2 in enumerate(metrics_to_compare[i+1:], i+1):
            plt.figure(figsize=(10, 6))
            plt.scatter(data[metric1], data[metric2], c=data["Noise_level"], cmap='viridis', alpha=0.7)
            plt.colorbar(label='Noise Level')
            plt.xlabel(metric1)
            plt.ylabel(metric2)
            plt.title(f'Relationship between {metric1} and {metric2}')
            plt.grid(True)
            plt.savefig(os.path.join(output_dirpath, f'{metric1}_vs_{metric2}.png'))
            plt.close()

    # Plot SNR metrics against noise at different contrast levels
    contrast_levels = sorted(data["Contrast_level"].unique())
    for metric in ["SNR4", "SNR5"]:
        plt.figure(figsize=(12, 8))
        for contrast in contrast_levels:
            subset = data[data["Contrast_level"] == contrast]
            x_values = np.log(subset["Noise_level"]) if log_scale else subset["Noise_level"]
            plt.plot(x_values, subset[metric], 'o-', label=f'Contrast={contrast}')


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
        "--set_index",type=int,required=False,
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
        print("INFO: created root output folder = ", root_dirpath)

    # output_dirpath = os.path.join(root_dirpath, "2d_data_quality_graphs")
    # if not os.path.exists(output_dirpath):
    #     os.mkdir(output_dirpath)
    #     print("INFO: created output folder = ", output_dirpath)

    plot_2d_data_quality(csv_path, root_dirpath, log_scale=True)


    plot_3d_data_quality(csv_path, root_dirpath, log_scale=True)

    # this method is for pair-wise comparison
    output_dirpath = os.path.join(root_dirpath, "2d_dataquality_compgraphs")
    if not os.path.exists(output_dirpath):
        os.mkdir(output_dirpath)
        print("INFO: created output folder = ", output_dirpath)
    plot_snr_vs_metrics(csv_path, output_dirpath, log_scale=True)


    # --csv_filepath
    # C:\PeterB\Projects\TestData\SEM_simulated_noise_contrast\artimagen_sim_noise_contrast_corrected\test\results_set1.csv
    # --output_dirpath
    # C:\PeterB\Projects\TestData\SEM_simulated_noise_contrast\artimagen_sim_noise_contrast_corrected\test

    # --csv_filepath
    # C:\PeterB\Presentations\NISTCollab\AndrasVladar\AI_halucinations\SPIE_conference\presentation\plots_2025-02-21\snr_propertieslist.csv
    # --output_dirpath
    # C:\PeterB\Presentations\NISTCollab\AndrasVladar\AI_halucinations\SPIE_conference\presentation


if __name__ == '__main__':
    main()

