#CSC 466 Fall 2023 - Lab 3: Decision Trees, part 1
#Othilia Norell and Soren Paetau \\ onorell@calpoly.edu  / spaetau@calpoly.edu

#HOW TO RUN:
#validation.py TrainingFile.csv restrictions.txt n



import sys
import pandas as pd
import json
import numpy as np
import classify
import InduceC45

"""

"""
def cross_val(df, class_var, n): #df
    indices = np.arange(df.shape[0])
    np.random.shuffle(indices)
    nocross = False
    if n == -1:
        folds = np.array_split(indices, len(indices))
    elif n == 0:
        nocross = True 
        folds = np.array_split(indices, 1)

    else:
        folds = np.array_split(indices, n) #k folds for cross validation
    
    threshold = 0.01 #change
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
            train = df.drop(fold).reset_index(drop=True)
        #print("in fold", i)
        tree =  InduceC45.get_tree(train, test_cols, threshold) #returns dict tree
        #print("tree ", i, " obtained", tree)
        classify.initialize_global(class_var, True, True) #1st True = is_training since doc asserts working with training file
        predictions = classify.generate_preds(test, tree)[0] #returns
        #print("preds generated")
        y_pred = pd.Series(predictions)
        y_actu = test[class_var]

        df_confusion = pd.crosstab(y_actu, y_pred,rownames=['Actual'], colnames=['Predicted'] )
        df_confusion = df_confusion.reindex(index = dom, columns= dom, fill_value = 0)
        #rint("fold", i, "\n", df_confusion)

        dfs.append(df_confusion)
        #train the model
        i += 1

    result = dfs[0]
    #print("res"+str(result)) 
    if len(dfs) > 1:
        for temp in dfs[1:]:
            result += temp

    #result = pd.concat([result, row_tot.rename('Row Total')], axis=1)
    #result = pd.concat([result, col_tot.rename('Column Total')])
    result.loc['Row Total']= result.sum()
    result['Col Total'] = result.sum(axis=1)

    return result



"""
"""
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

"""
Calculates and returns the accuracy, precision and recall
"""
def metrics(cross_ret):
    #ACCURACY 
    # = (TP + TN) / (TP + TN + FP + FN)
    total_conf_matrix_array = cross_ret.to_numpy() #converting the confusion matrix to a numpy array
    conf_matrix_array = total_conf_matrix_array[:-1, :-1] #excluding the total rows/columns
    TP = np.diag(conf_matrix_array) #
    accuracy = TP.sum() / conf_matrix_array.sum()
    print("")
    print("Accuracy: " +str(accuracy*100) + "%")

    #PRECISION & RECALL 
    # --> fine for this assignment, report it for one of the classes (?)
    precision = {} #Precision = TP / (TP + FP) --> vertically
    recall = {}  #Recall = TP / (TP + FN) --> Horisontolly
    class_names = cross_ret.index.tolist()[:-1]  # Exclude the 'Row Total' label

    for j in range(len(TP)): 
        nom = TP[j] #TP
        denom_precision = total_conf_matrix_array[-1][j] #sum of column (is found in the total)

        if denom_precision != 0: 
            precision[class_names[j]] = "Precision for " + str(class_names[j]) + " = " + str((nom/denom_precision)*100) + "%"
        else: 
            precision[class_names[j]] = "Precision for " + str(class_names[j]) + " = " + str(0) + "%"
        
        denom_recall = total_conf_matrix_array[j][-1]
        if denom_recall != 0:
            recall[class_names[j]] = "Recall for " + str(class_names[j]) + " = " + str((nom/denom_recall)*100) + "%"
        else: 
            recall[class_names[j]] = "Recall for " + str(class_names[j]) + " = " + str(0) + "%"


    print(precision)
    print(recall) 


"""
Main Function
"""
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
    #print("global initialized")
 #1st True = is_training since doc asserts working with training file
    #2nd true is silent since we don't want outputs, should be the case
  
    cross_ret = cross_val(D, class_var, n)
    sys.stdout.write(cross_ret.to_string())
    metrics(cross_ret)


if __name__ == "__main__":
    main()

    
