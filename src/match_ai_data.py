'''
NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
'''

# Author: Peter Bajcsy
# Date: 2023-04-01
# Description: This script matches AI model evaluation metrics with data quality metrics by merging CSV files based on a common column.

import argparse
import sys

import pandas as pd


def match_csv_files(file1_path, file2_path, match_column):
    """
    Load two CSV files and match rows based on a specified column.

    Parameters:
    -----------
    file1_path : str
        Path to the first CSV file
    file2_path : str
        Path to the second CSV file
    match_column : str
        Name of the column to match on

    Returns:
    --------
    pandas.DataFrame
        Merged dataframe containing matched rows
    """
    # Load the CSV files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Check if the match column exists in both dataframes
    if match_column not in df1.columns or match_column not in df2.columns:
        raise ValueError(f"Match column '{match_column}' not found in '{file1_path}'' or '{match_column}' not found in '{file2_path}'.")

    # Merge the dataframes on the specified column
    # Using suffixes to distinguish columns with the same name
    merged_df = pd.merge(df1, df2,
                         on=match_column,
                         how='inner',  # inner join keeps only matching rows
                         suffixes=('_file1', '_file2'))

    print(f"Found {len(merged_df)} matching rows based on '{match_column}'")
    return merged_df


# Example usage
def main():
    parser = argparse.ArgumentParser(description="match the Dice index and confusion matrix from the AI model evaluation with the training data quality metrics")
    parser.add_argument(
        "--input_ai_model", type=str, required=True,
        help="Path to the CSV file with  AI model evaluations."
    )
    parser.add_argument(
        "--input_data_quality", type=str, required=True,
        help="Path to teh CSV file with image quality metrics."
    )
    parser.add_argument(
        "--output_filepath", type=str, required=True,
        help="path to a CSV file with merged entries.",
        default=f"../matched_results.csv"
    )

    args = parser.parse_args()
    if args.input_ai_model is None:
        print('ERROR: missing input_ai_model CSV file')
        return

    if args.input_data_quality is None:
        print('ERROR: missing input_data_quality CSV file ')
        return

    input_ai_model_csv = args.input_ai_model
    input_data_quality_csv = args.input_data_quality
    output_filepath = args.output_filepath

    # if not os.path.exists(output_folder):
    #     # create the output folder
    #     os.mkdir(output_folder)
    #     print("INFO: created output folder = ", output_folder)

    print('Arguments:')
    print('input_ai_model_csv= {}'.format(input_ai_model_csv))
    print('input_data_quality= {}'.format(input_data_quality_csv))
    print('output filepath = {}'.format(output_filepath))


    # match_column_ai_model = "IMAGE-NAME"  # Replace with your column name
    # match_column_data_quality = "id"
    match_column = "IMAGE-NAME"  # Replace with your column name
    try:
        # Match the files
        #result = match_csv_files(input_ai_model_csv, input_data_quality_csv, match_column_ai_model, match_column_data_quality)
        result = match_csv_files(input_ai_model_csv, input_data_quality_csv, match_column)

        # Display the first few rows of the result
        print("\nFirst few matched rows:")
        print(result.head())

        # Optionally save the result to a new CSV file
        result.to_csv(output_filepath, index=False)

    except Exception as e:
        print(f"Error: {e}")



if __name__ == "__main__":
    print('Python %s on %s' % (sys.version, sys.platform))
    #sys.path.extend([WORKING_DIR_AND_PYTHON_PATHS])

    main()