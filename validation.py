import sys
import pandas as pd
import json
import numpy as np

allbutone = False
nocross = False

#validation.py TrainingFile.csv restrictions.txt n
# which will take as input the training file, the optional restrictions file and an integer number n

def parser(filename):
    pass

if sys.argv[3] == -1:
    allbutone = True
elif sys.argv == 0:
    nocross = True
