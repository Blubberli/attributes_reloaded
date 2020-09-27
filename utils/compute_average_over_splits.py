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
    parameter = None
    for line in open(log_file, "r"):
        if "{" in line and "}" in line:
            matcher = re.compile("{.*}")
            parameter = matcher.search(line).group(0)
            parameter = ast.literal_eval(parameter)
        if "f1 score" in line:
            f1_macro = line.split(";")[1].strip().split(",")[0].strip().replace("f1 score macro: ", "")
            f1_weighted = line.split(";")[1].strip().split(",")[1].strip().replace("f1 score weighted : ", "")
        if "precision at rank 1" in line:
            precision_1 = line.split(":")[2].split(";")[0].strip()
    if f1_macro and precision_1 and f1_weighted:
        return float(f1_macro), float(f1_weighted), float(precision_1), parameter
    else:
        print("invalid log file")


def get_parameter_value(parameter_name, parameter_dic):
    """
    Get the value of a parameter of interest
    :param parameter_name: the parameter, the current value shall be extracted for
    :param parameter_dic: the dictionary containing the parameter for a log file
    :return: the value for the parameter of interest if available
    """
    for par_name, par_val in parameter_dic.items():
        if parameter_name in par_name:
            return par_val
        if type(par_val) == dict:
            for sub_parameter_name, sub_parameter_val in par_val.items():
                if parameter_name in sub_parameter_name:
                    return sub_parameter_val
    else:
        return "parameter not found"


def get_average(dir_1, parameter_name):
    """
    Compute the average F1 score and precision at rank 1 for two directories of log files. Each directory should
    contain the log files for a split, exactly with the same number of log files and the same range of parameter values.
    The parameter name specifies which parameter the result is reported for, e.g. 'dropout' or 'transformations'
    :param dir_1: the path to the directory containing all log files for the first split
    :param dir_2: the path to the directory containing all log files for the second split
    :param parameter_name: the name of the parameter that is tuned
    """
    log_files_1 = os.listdir(dir_1)
    parameter2f1_macro = defaultdict(list)
    parameter2f1_weighted = defaultdict(list)
    parameter2prec = defaultdict(list)
    for f in log_files_1:
        path = dir_1  + f
        print(path)
        f1_macro, f1_weighted, prec1, parameter = read_log_file(path)
        parameter_val = get_parameter_value(parameter_name, parameter)
        parameter2f1_macro[parameter_val].append(f1_macro)
        parameter2f1_weighted[parameter_val].append(f1_weighted)
        parameter2prec[parameter_val].append(prec1)
    for par_val, values in parameter2prec.items():
        result = np.average(np.array(values))
        print(parameter_name + ": %s" % str(par_val))
        print("precision at rank 1 : %.3f" % result)
    for par_val, values in parameter2f1_macro.items():
        result = np.average(np.array(values))
        print(parameter_name + ": %s" % str(par_val))
        print("F1 macro : %.3f" % result)
    for par_val, values in parameter2f1_weighted.items():
        result = np.average(np.array(values))
        print(parameter_name + ": %s" % str(par_val))
        print("F1 weighted : %.3f" % result)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("dir_1", help="path to the directory that contains log files for the first split")
    argp.add_argument("parameter_name", help="name of the parameter that is tuned, e.g 'dropout' or 'transformations'")
    argp = argp.parse_args()
    get_average(argp.dir_1, argp.parameter_name)
