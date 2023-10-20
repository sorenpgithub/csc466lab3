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



def cross_val(df, k, class_var):
    
    indices = np.arrange(df.shape[0])
    if n == -1:
        allbutone = True
    elif n == 0:
        nocross = True #FIXXXXXXXXXXXX
    else:
        folds = np.array_split(indices, k) #k folds for cross validation
    threshold = 0.1 #change
    dfs = []
    dim = df["class_var"].nunique()
    conf_matrix = pd.DataFrame(nrow = dim + 1, ncol = dim + 1) #define correct dimensions!!!!!!!

    for fold in folds:
        test = df.iloc[fold]
        train = df.iloc[-fold]
        tree =  InduceC45.foobar(train, class_var, threshold) #returns dict tree
        predictions = classify.generate_preds(test, class_var, tree)[0] #returns
        y_pred = pd.Series(predictions)
        y_actu = test[class_var]
        df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
        dfs.append(df_confusion)
        #train the model
    for temp in dfs: #in theory adds together matrices
        conf_matrix = conf_matrix + temp

def main()
    n = 5
    restfile = None #default vals for optional params
    if len(sys.argv) >= 3: #assigning params if input
        if sys.argv[2] != "None":
            restfile = sys.argv[2]
        if len(sys.argv) == 4:
            n = int(sys.argv[3])
    ret = InduceC45.parser(sys.argv[1], restfile)
    D = ret[0]
    class_var = ret[1]

if __name__ == "__main__":
    main()





    
    
