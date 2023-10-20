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



def cross_val(df, class_var, n): #df
    
    indices = np.arange(df.shape[0])
    np.random.shuffle(indices)
    nocross = False
    allbutone = False
    if n == -1:
        allbutone = True
    elif n == 0:
        nocross = True #FIXXXXXXXXXXXX
    else:
        folds = np.array_split(indices, n) #k folds for cross validation
    
    threshold = 0.1 #change
    dfs = []
    dom = df[class_var].unique()
     #define correct dimensions!!!!!!!
    test_cols = list(df.columns) #column names
    test_cols.remove(class_var)
    i= 0 
    for fold in folds:
        test = df.iloc[fold].reset_index(drop = True)

        if nocross:
            train = test
        else:
            train = df.iloc[-fold]
        #print(train, class_var, threshold)
        
        tree =  InduceC45.get_tree(train, test_cols, threshold) #returns dict tree
        classify.initialize_global(class_var, True, True) #1st True = is_training since doc asserts working with training file
        #print("fold", i, "\n", tree)
        predictions = classify.generate_preds(test, tree)[0] #returns

        y_pred = pd.Series(predictions)
        y_actu = test[class_var]
        #print("pred: fold", i, "\n", y_pred)
        #print("actu: fold", i, "\n", y_actu)
        df_confusion = pd.crosstab(y_actu, y_pred,rownames=['Actual'], colnames=['Predicted'] )
        df_confusion = df_confusion.reindex(index = dom, columns= dom, fill_value = 0)
        print("fold", i, "\n", df_confusion)

        dfs.append(df_confusion)
        #train the model
        i += 1


  
    result = dfs[0] 
    if len(dfs) > 1:
        for temp in dfs[1:]:
            result += temp
    row_tot = result.sum(axis=1)
    col_tot = result.sum()

    #result = pd.concat([result, row_tot.rename('Row Total')], axis=1)
    #result = pd.concat([result, col_tot.rename('Column Total')])
    result.loc['Row Total']= result.sum()
    result['Col Total'] = result.sum(axis=1)


    return result


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
 #1st True = is_training since doc asserts working with training file
    #2nd true is silent since we don't want outputs, should be the case
  
    cross_ret = cross_val(D, class_var, n)
    print(cross_ret.to_string())


if __name__ == "__main__":
    main()





    
    
