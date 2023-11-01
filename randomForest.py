#python3 randomForest.py training.csv numAttribute numDatapoints numTrees
import sys
import pandas as pd
import numpy as np
import InduceC45
import evaluate
import random

# python3 randomForest.py adult+stretch.csv 3 4 50
"""
Randomizes data based on intuitive parameters
Uses pandas
"""
def rand_data(D, class_var, numAtt, numData): #D is pandas DF, rest is defined
    df = D.sample(numData, replace = True)
    #print(df)
    cols = list(D.columns)
    cols.remove(class_var)
    newcols = random.sample(cols, numAtt)
    newcols.append(class_var)
    df = df[newcols]
    return df


def main():
    path = sys.argv[1]
    restfile = None #change if desired
    ret = InduceC45.parser(path, restfile)
    D = ret[0]
    class_var = ret[1]
    numAttributes = int(sys.argv[2])
    numDataPoints = int(sys.argv[3])
    numTrees = int(sys.argv[4])
    InduceC45.initialize_global(path, restfile, False) #may not work but needed for evaluate.crossval to run
    folds = 10#should be fixed to 10
    conf = evaluate.cross_val(D, class_var, folds, False, [numTrees, numAttributes, numDataPoints]) #df, class_var, n, silent, numTrees
    #print(conf[0])
    outs = evaluate.metrics(conf)
    evaluate.output(outs)
    preds_df = conf[2]
    name = '.'.join(path.split(".")[:-1]) + ".forest.results.csv"
    preds_df.to_csv(name, index=False, header = False)



if __name__ == "__main__":
    main()