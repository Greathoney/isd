# ID: ELEC946001
# NAME: Intelligent System Design
# File name: read_iris_v1.py
# Platform: Python 3.8.5 on Windows 10
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
