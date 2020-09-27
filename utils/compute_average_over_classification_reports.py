import os
from collections import defaultdict
import pandas as pd
import argparse
import numpy as np


def read_classification_report(path):
    report = pd.read_csv(path, sep="\t")
    macro_f1 = None
    weighted_f1 = None
    att2f1 = {}
    for ix, row in report.iterrows():
        att = row[0]
        f1 = row[3]
        if "macro avg" in att:
            macro_f1 = float(row[3])
        if "weighted avg" in att:
            weighted_f1 = float(row[3])
        else:
            att2f1[att] = f1
    return att2f1, macro_f1, weighted_f1


def get_average(dir_1, save_path):
    """
    Compute the average F1 score and precision at rank 1 for two directories of log files. Each directory should
    contain the log files for a split, exactly with the same number of log files and the same range of parameter values.
    The parameter name specifies which parameter the result is reported for, e.g. 'dropout' or 'transformations'
    :param dir_1: the path to the directory containing all log files for the first split
    :param dir_2: the path to the directory containing all log files for the second split
    :param parameter_name: the name of the parameter that is tuned
    """
    log_files_1 = os.listdir(dir_1)
    f1_macro_scores = []
    f1_weighted_scores = []
    file = open(save_path + "_overall_average.txt", "w")
    attribute_file = open(save_path + "_attribute_average.csv", "w")
    att2f1_scores = defaultdict(list)
    for f in log_files_1:
        path = dir_1 + f
        print(path)
        att2f1, macro_f1, weight_f1 = read_classification_report(path)
        for attribute, f1score in att2f1.items():
            att2f1_scores[attribute].append(f1score)
        f1_macro_scores.append(macro_f1)
        f1_weighted_scores.append(weight_f1)

    file.write("F1 macro : %.3f\n" % np.average(np.array(f1_macro_scores)))
    file.write("F1 weighted : %.3f" % np.average(np.array(f1_weighted_scores)))
    file.close()

    for attribute, f1 in att2f1_scores.items():
        attribute_file.write("%s\t%.3f\n" % (attribute, np.average(np.array(f1))))
    attribute_file.close()


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("dir_1", help="path to the directory that contains log files for the first split")
    argp.add_argument("save_path")
    argp = argp.parse_args()
    get_average(argp.dir_1, argp.save_path)
