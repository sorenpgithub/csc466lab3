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
    accuracies = []
    dom = df[class_var].unique()
    #define correct dimensions!!!!!!!
    test_cols = list(df.columns) #column names
    test_cols.remove(class_var)
    i = 0 

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
        predictions = classify.generate_preds(test, tree) #returns
        #print("preds generated")
        y_pred = pd.Series(predictions[0])
        y_actu = test[class_var]
        
        
        count_correct = predictions[1] #may not work, but has been somewhat tested
        accu = count_correct / len(y_pred) #proportion of correct
        accuracies.append(accu)


        df_confusion = pd.crosstab(y_actu, y_pred,rownames=['Actual'], colnames=['Predicted'] )
        df_confusion = df_confusion.reindex(index = dom, columns= dom, fill_value = 0)
        #rint("fold", i, "\n", df_confusion)

        dfs.append(df_confusion)
        #train the model
        i += 1

    conf_mat = dfs[0]
    #print("res"+str(result)) 
    if len(dfs) > 1:
        for temp in dfs[1:]:
            conf_mat += temp

    #result = pd.concat([result, row_tot.rename('Row Total')], axis=1)
    #result = pd.concat([result, col_tot.rename('Column Total')])
    conf_mat.loc['Row Total']= conf_mat.sum()
    conf_mat['Col Total'] = conf_mat.sum(axis=1)
    
    return (conf_mat, np.mean(accuracies))



"""
"""
def output(temp):
    for out in temp:
        print(out)

"""
Calculates and returns the accuracy, precision and recall
"""
def metrics(cross_ret): #(conf_matrix, mean accuracies)
    #ACCURACY 
    # = (TP + TN) / (TP + TN + FP + FN)
    conf = cross_ret[0]
    total_conf_matrix_array = conf.to_numpy() #converting the confusion matrix to a numpy array
    conf_matrix_array = total_conf_matrix_array[:-1, :-1] #excluding the total rows/columns
    TP = np.diag(conf_matrix_array) #
    accuracy = TP.sum() / conf_matrix_array.sum()
    temp = []
    temp.append(conf.to_string())
    temp.append("Overall Accuracy: " +str(accuracy*100) + "%")
    temp.append("Average Accuracy: " + str(cross_ret[1]*100) + "%")

    #PRECISION & RECALL 
    # --> fine for this assignment, report it for one of the classes (?)
    precision = {} #Precision = TP / (TP + FP) --> vertically
    recall = {}  #Recall = TP / (TP + FN) --> Horisontolly
    class_names = conf.index.tolist()[:-1]  # Exclude the 'Row Total' label

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


    temp.append(precision)
    temp.append(recall) 


    return temp


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
    outs = metrics(cross_ret)
    output(outs)


if __name__ == "__main__":
    main()

    
