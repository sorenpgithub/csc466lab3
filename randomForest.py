#python3 randomForest.py numAttribute numDatapoints numTrees
import sys
import pandas as pd
import json
import numpy as np

import evaluate
import random

def parser(path):
    pass

def rand_data(D, class_var): #D is pandas DF, rest is defined
    df = D.sample(numDataPoints, replace = True)
    cols = D.columns
    cols.remove(class_var)
    newcols = random.sample(cols, numAttributes)
    newcols.append(class_var)
    df = df[newcols]
    return df

def main():
    global class_var, numAttributes, numDataPoints
    ret = parser
    D = ret[0]
    class_var = [1]
    numAttributes = sys.argv[2]
    numDataPoints = sys.argv[3]
    numTrees = sys.argv[4]
    evaluate.cross_val(D, class_var, 10, False, numTrees) #df, class_var, n, silent, numTrees


if __name__ == "__main__":
    main()