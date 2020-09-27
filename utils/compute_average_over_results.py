import ast
import os
import re
from collections import defaultdict
import argparse
import numpy as np


def read_log_file(log_file):
    """
    Reads in a log file and stores the parameter in a dictionary
    :param log_file: path to a log file
    :return: the f1 score and precision at rank 1 and the dictionary containing the parameter the model was trained with
    """
    f1_macro = None
    f1_weighted = None
    precision_1 = None
    precision_3 = None
    for line in open(log_file, "r"):
        if "f1 score" in line:
            f1_macro = line.split(";")[1].strip().replace("f1 score macro: ", "")
            f1_weighted = line.split(";")[2].strip().replace("f1 score weighted:", "")
        if "precision at rank 1" in line:
            precision_1 = line.split(":")[1].strip()
        if "precision at rank 3" in line:
            precision_3 = line.split(":")[1].strip()
    if f1_macro and precision_1 and f1_weighted:
        return float(f1_macro), float(f1_weighted), float(precision_1), float(precision_3)
    else:
        print("invalid log file")


def get_average(dir_1):
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
    prec1_scores = []
    prec3_scores = []
    for f in log_files_1:
        path = dir_1  + f
        print(path)
        f1_macro, f1_weighted, prec1, prec3 = read_log_file(path)
        f1_macro_scores.append(f1_macro)
        f1_weighted_scores.append(f1_weighted)
        prec1_scores.append(prec1)
        prec3_scores.append(prec3)

    print("precision at rank 1 : %.3f" % np.average(np.array(prec1_scores)))
    print("precision at rank 3 : %.3f" % np.average(np.array(prec3_scores)))
    print("F1 macro : %.3f" % np.average(np.array(f1_macro_scores)))
    print("F1 weighted : %.3f" % np.average(np.array(f1_weighted_scores)))


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("dir_1", help="path to the directory that contains log files for the first split")
    argp = argp.parse_args()
    get_average(argp.dir_1)
