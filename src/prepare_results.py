import os
import glob
import shutil
import argparse

"""
This script runs over a batch of results and combines them from separate folders to one 
the folder structure SHOULD be something like the following:

<ROOT OF RESULT BATCH>
    |   
    YYYY-MM-DD_0/
    |       |
    |       |__HH-MM-SS_0/
    |       |   |__0.csv
    |       |   |__0_scores.csv
    |       |   |__0_scores.txt
    |       |   |__revamp.log
    |       |
    |       |__HH-MM-SS_1/
    |           |__1.csv
    |           |__1_scores.csv
    |           |__1_scores.txt
    |           |__revamp.log        
    |   
    YYYY-MM-DD_1/
            |
            |__HH-MM-SS_0/
                |__2.csv
                |__2_scores.csv
                |__2_scores.txt
                |__revamp.log

It doesn't matter how many result runs are in a particular day, e.g., there could be seven different HH-MM-SS_n folders under one YYYY-MM-DD folder.
- This script checks for duplicate-named files
- This script combines all .log files into a single log file in the order of their [PASS] number

"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser( \
        description='Example script with default values' \
        ,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--input-dir", help="Directory of results", type=str, default=dir, required=True)
    args = parser.parse_args()

    root_path = args.input_dir
    # root_path = "results/car/test_results"

    n=7
    # # Get all subfolders
    sub_folders = glob.glob(os.path.join(root_path, "*/*"))

    # Check for duplicates
    files_seen = set()
    for sub_folder in sub_folders:
        # Check only for files other than .log files
        files = glob.glob(os.path.join(sub_folder, "[!run]*"))
        for file in files:
            if os.path.basename(file) not in files_seen:
                files_seen.add(os.path.basename(file))
            else:
                print(f"Error: Duplicate file found in {sub_folder}")
                exit(1)

    # Loop through all csv files in the root folder and its subdirectories
    for csv_file_path in glob.glob(os.path.join(root_path, "**", "*.csv"), recursive=True):
        # Check if this is an n.csv file
        file_name = os.path.basename(csv_file_path)
        if file_name[:-4].isdigit() and int(file_name[:-4]) <= n:
            # Set the sub-folder path and the file names
            sub_folder_path = os.path.dirname(csv_file_path)
            i = int(file_name[:-4])
            file_path = os.path.join(sub_folder_path, file_name)
            scores_csv_path = os.path.join(sub_folder_path, f"{i}_scores.csv")
            scores_txt_path = os.path.join(sub_folder_path, f"{i}_scores.txt")
            
            # Check if the ".hydra" folder exists in the same subfolder as 0.csv
            if i == 0 and os.path.exists(os.path.join(sub_folder_path, ".hydra")):
                hydra_folder_path = os.path.join(sub_folder_path, ".hydra")
                shutil.copytree(hydra_folder_path, os.path.join(root_path, ".hydra"))
            
            # Rename the run.log file to run_n.log
            log_file_path = os.path.join(sub_folder_path, "revamp.log")
            new_log_file_path = os.path.join(sub_folder_path, f"revamp_{i}.log")
            os.rename(log_file_path, new_log_file_path)
            
            # Copy the files to the root folder
            shutil.copy(file_path, root_path)
            shutil.copy(new_log_file_path, root_path)
            shutil.copy(scores_csv_path, root_path)
            shutil.copy(scores_txt_path, root_path)

    # Append the contents of all revamp_n.log files (where n is not 0) to revamp_0.log
    run_n_log_files = glob.glob(os.path.join(root_path, "revamp_*.log"))
    run_n_log_files.sort()
    for log_file_path in run_n_log_files:
        log_file_name = os.path.basename(log_file_path)
        if log_file_name.startswith("run_") and log_file_name != "revamp_0.log":
            with open(log_file_path, "r") as log_file:
                with open(os.path.join(root_path, "revamp_0.log"), "a") as run_0_file:
                    run_0_file.write(log_file.read())
            # Remove the copied log file
            os.remove(log_file_path)

    # Remove all subfolders
    for subfolder in glob.glob(os.path.join(root_path, "*", "")):
        shutil.rmtree(subfolder)
