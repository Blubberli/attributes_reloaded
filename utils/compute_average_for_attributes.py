import pandas as pd
import os
from collections import defaultdict
import numpy as np
import argparse


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


def read_rank_report(path):
    report = pd.read_csv(path, sep="\t")
    att2p1 = {}
    att2p3 = {}
    for ix, row in report.iterrows():
        att = row[0]
        p1 = row[1]
        p3 = row[2]
        att2p1[att] = p1
        att2p3[att] = p3
    return att2p1, att2p3


def get_average(dir_1, dir_2, save_path):
    """
    Compute the average F1 score and precision at rank 1 for two directories of log files. Each directory should
    contain the log files for a split, exactly with the same number of log files and the same range of parameter values.
    The parameter name specifies which parameter the result is reported for, e.g. 'dropout' or 'transformations'
    :param dir_1: the path to the directory containing all log files for the first split
    :param dir_2: the path to the directory containing all log files for the second split
    :param parameter_name: the name of the parameter that is tuned
    """
    log_files_1 = os.listdir(dir_1)
    attribute2f1 = defaultdict(list)

    f1_macro_scores = []
    f1_weighted_scores = []
    for f in log_files_1:
        path = dir_1 + f
        att2f1, macro_f1, weighted_f1 = read_classification_report(path)
        f1_macro_scores.append(macro_f1)
        f1_weighted_scores.append(weighted_f1)
        for attribute, f1 in att2f1.items():
            attribute2f1[attribute].append(f1)

    attribute2p1 = defaultdict(list)
    attribute2p3 = defaultdict(list)

    log_files_2 = os.listdir(dir_2)
    for f in log_files_2:
        path = dir_2 + f
        att2p1, att2p3 = read_rank_report(path)
        for attribute, p1 in att2p1.items():
            attribute2p1[attribute].append(p1)
            attribute2p3[attribute].append(att2p3[attribute])

    f = open(save_path
             + "averaged_attribute_scores.csv", "w")
    f.write("attribute\tp1\tp3\tf1\n")

    # print("F1 macro : %.3f" % np.average(np.array(f1_macro_scores)))
    # print("F1 weighted : %.3f" % np.average(np.array(f1_weighted_scores)))
    f.write("F1 macro\t%.2f\t\n" % np.average(np.array(f1_macro_scores)))
    f.write("F1 weighted\t%.2f\t\n" % np.average(np.array(f1_weighted_scores)))
    for attribute, f1scores in attribute2f1.items():
        # print("%s : %.3f" % (attribute, np.average(np.array(f1scores))))
        p1 = attribute2p1[attribute]
        p3 = attribute2p3[attribute]
        if "f1" in attribute:
            continue
        f.write("%s\t%.2f\t%.2f\t%.2f\n" % (
            attribute, np.average(np.array(p1)), np.average(np.array(p3)), np.average(np.array(f1scores))))
    f.close()


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("dir_1", help="path to the directory that contains log files for the first split")
    argp.add_argument("dir_2", help="if specified, compute rank-based measures for each attribute")
    argp.add_argument("save_path", help="path to the directory to store the results")
    argp = argp.parse_args()

    get_average(argp.dir_1, argp.dir_2, argp.save_path)
