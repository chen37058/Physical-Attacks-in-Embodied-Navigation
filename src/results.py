"""
This script processes raw outputs from attack_dt2.py
It extracts the losses and outputs all values to a .csv
matching the input filename
usage: python src/results.py -i results/results.txt
output: results.csv
"""
import re
import csv
import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input file path.")
    args = parser.parse_args()
    
    fp = args.input

    # get number of passes, make individual .csvs
    with open(fp, "r") as log:
        content = log.read()
    passes = re.findall(r"\[PASS \d+\]", content)
    num_passes = len(set(passes))    

    pass_names = []
    for p in passes:
        result = re.search(r"\[PASS (\d+)\]", p)
        number = int(result.group(1))
        pass_names.append(number)
    pass_names = sorted(list(set(pass_names)))

    for p in range(num_passes):
        base = pass_names[p]
        dir = os.path.dirname(fp)
        op = os.path.join(dir,f"{base}.csv")

        losses = []
        with open(fp, 'r') as file:           
            for line in file:
                if f"[PASS {base}]" in line:
                    if "loss" in line:
                        result = re.search(r'loss: (\d+\.\d+)', line)
                        if result:
                            extracted_number = float(result.group(1))
                            losses.append(extracted_number)
                            print(extracted_number)
                        else:
                            print("Number not found in string.") 
        
        with open(op, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(losses)
