# ID: 2018116323 (undergraduate)
# NAME: DaeHeon Yoon
# File name: read_iris_v2.py
# Platform: Python 3.8.8 on Windows 10
# Required Package(s): sys numpy

# 2018116323 윤대헌

##############################################################
# Template file for homework programming assignment 1
# Modify the first 5 lines according to your implementation
# This file is just for an example. Feel free to modify it.
##############################################################

##############################################################
# NOTE: import sys and numpy only. 
# No other packages are allowed to be imported
##############################################################
import sys
import numpy as np

if len(sys.argv) < 2:
    print('usage: ' + sys.argv[0] + ' text_file_name')
else:
    ##############################################################
    # WRITE YOUR OWN CODE LINES
    # - open the input file, without pandas or csv packages
    # - read header line
    # - read data and class labels
    # - compute mean and standard deviation
    # - disply them 
    ##############################################################
    
    if sys.argv[1][-3:].lower() == 'csv': delimeter = ','
    else: delimeter = '\t'  # default is all white spaces 

    f = open(sys.argv[1], 'r')

    print(sys.argv[1], delimeter)

    col_title = f.readline()
    col_title = col_title.split(delimeter)
    col_title = col_title[:-1]

    data = [np.array([]) for i in range(len(col_title))]

    while True:
        line = f.readline()
        if not line:
            break

        line = line.split(delimeter)
        line = line[:-1]
        for index in range(len(col_title)):
            data[index] = np.append(data[index], float(line[index]))

    f.close()

    print("------------------------------------------------------------------")

    print("%-6s" % "", end='')
    for index in col_title:
        print("%15s" % index, end="")
    print()

    print("%-6s" % "mean", end="")
    for index in range(len(col_title)):
        mean = np.mean(data[index])
        mean = np.round(mean, 2)
        print("%15.02f" % mean, end="")
    print()

    print("%-6s" % "std", end="")
    for index in range(len(col_title)):
        std = np.std(data[index])
        std = np.round(std, 2)
        print("%15.02f" % std, end="")
    print()
    print("------------------------------------------------------------------")