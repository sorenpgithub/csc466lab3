import sys
import pandas as pd
import json
import numpy as np
import classify
import InduceC45

allbutone = False
nocross = False

#validation.py TrainingFile.csv restrictions.txt n
# which will take as input the training file, the optional restrictions file and an integer number n

#INIT



def cross_val(df, class_var, n):
    
    indices = np.arrange(df.shape[0])
    if n == -1:
        allbutone = True
    elif n == 0:
        nocross = True #FIXXXXXXXXXXXX
    else:
        folds = np.array_split(indices, n) #k folds for cross validation
    
    threshold = 0.1 #change
    dfs = []
    dim = df["class_var"].nunique()
    conf_matrix = pd.DataFrame(nrow = dim + 1, ncol = dim + 1) #define correct dimensions!!!!!!!

    for fold in folds:
        test = df.iloc[fold]
        if nocross:
            train = test
        else:
            train = df.iloc[-fold]
        
        tree =  InduceC45.foobar(train, class_var, threshold) #returns dict tree
        predictions = classify.generate_preds(test, class_var, tree)[0] #returns

        y_pred = pd.Series(predictions)
        y_actu = test[class_var]
        df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

        dfs.append(df_confusion)
        #train the model
    for temp in dfs: #in theory adds together matrices
        conf_matrix = conf_matrix + temp #might break
  
    return conf_matrix

def output(temp):
    pass
    # #https://stackoverflow.com/questions/69916525/easy-way-to-extract-common-measures-such-as-accuracy-precision-recall-from-3x3
    # cm = conf_matrix.to_numpy()
    # diag = cm.diagonal()[:-1]

    # accuracy  = diag / cm[ -1, -1]
    # precision = diag / cm[  dim,:-1]
    # recall    = diag / cm[:-1,  dim]
    # f_score   = 2 * precision * recall / (precision + recall)

    # out = pd.DataFrame({'Accuracy': accuracy, 
    #                     'Precision': precision, 
    #                     'Recall': recall,
    #                     'F-score': f_score}).round(2)

def main():
    n = 5
    restfile = None #default vals for optional params
    if len(sys.argv) >= 3: #assigning params if input
        if sys.argv[2] != "None":
            restfile = sys.argv[2]
        if len(sys.argv) == 4:
            n = int(sys.argv[3])
    path = sys.argv[1]
    ret = InduceC45.parser(path, restfile)
    D = ret[0]
    class_var = ret[1]
    InduceC45.initialize_global(path, restfile, False)
    classify.initialize_global(D, class_var, True, True) #1st True = is_training since doc asserts working with training file
    #2nd true is silent since we don't want outputs, should be the case
  
    cross_ret = cross_val(D, class_var, n)
    cross_ret


if __name__ == "__main__":
    main()





    
    
