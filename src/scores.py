import csv
import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input file path.")
    parser.add_argument("-t", "--target", help="target class")
    args = parser.parse_args()
    fp = args.input
    target = args.target
    base = os.path.basename(fp)
    base = os.path.splitext(base)[0]
    dir = os.path.dirname(fp)
    op = os.path.join(dir,f"{base}.csv")


    with open(fp, "r") as file, open(op, "w", newline="") as csvfile:
        lines = file.readlines()
        writer = csv.writer(csvfile)
        for idx, line in enumerate(lines):
            if "Predicted Class:" in line:
                if target in line:
                    writer.writerow([idx, True])
                else:
                    writer.writerow([idx, False])
