#python3 randomForest.py numAttribute numDatapoints numTrees
import sys
import pandas as pd
import json
import numpy as np

import evaluate
import random

def parser(path):
    pass

def build_data(D, m, k, class_var): #D is pandas DF, rest is defined
    df = D.sample(k, replace = True)
    cols = D.columns
    cols.remove(class_var)
    newcols = random.sample(cols, m)
    newcols.append(class_var)
    df = df[newcols]
    return df

def main():
    ret = parser
    D = ret[0]
    class_var = [1]
    numAttributes = sys.argv[2]
    numDataPoints = sys.argv[3]
    numTrees = sys.argv[4]
    build_data(D, numAttributes, numDataPoints, class_var)
    evaluate(D, numTrees, numTrees, False)


if __name__ == "__main__":
    main()