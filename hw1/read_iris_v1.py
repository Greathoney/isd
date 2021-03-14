# ID: 2018116323 (undergraduate)
# NAME: DaeHeon Yoon
# File name: read_iris_v1.py
# Platform: Python 3.8.8 on Windows 10
# Required Package(s): sys numpy pandas

import sys
import numpy as np
import pandas as pd

if len(sys.argv) < 2:
    print('usage: ' + sys.argv[0] + ' text_file_name')
else:
    # determine delimieter based on file extension - may be used by pandas
    # this is just to show how to use command line arguments. 
    # any modification is accepted depending on your implementation.
    if sys.argv[1][-3:].lower() == 'csv': delimeter = ','
    else: delimeter = '[ \t\n\r]'  # default is all white spaces 

    # read CSV/Text file with pandas
    f = pd.read_csv(sys.argv[1],sep=delimeter,engine='python')
    

    ##############################################################
    # WRITE YOUR OWN CODE LINES
    # - read header line
    # - read data and class labels
    # - compute mean and standard deviation
    # - disply them 
    ##############################################################
    print("------------------------------------------------------------------")
    col_title = f.columns.tolist()

    print("%-6s" % "", end='')
    for index in col_title[:-1]:
        print("%15s" % index, end="")
    print()

    print("%-6s" % "mean", end="")
    for index in col_title[:-1]:
        mean = np.mean(f[index])
        mean = np.round(mean, 2)
        print("%15.02f" % mean, end="")
    print()

    print("%-6s" % "std", end="")
    for index in col_title[:-1]:
        std = np.std(f[index])
        std = np.round(std, 2)
        print("%15.02f" % std, end="")
    print()
    print("------------------------------------------------------------------")