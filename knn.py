#CSC 466 Fall 2023 - Lab 3: Decision Trees, part 1
#Othilia Norell and Soren Paetau \\ onorell@calpoly.edu  / spaetau@calpoly.edu

#HOW TO RUN: 
#knn.py dataset.csv K 
import sys
import pandas as pd
import numpy as np 
import json
import classify

# PARAMETERS TO CONSIDER: 
## no neighbors K
## distance metrics
## weights

"""

"""
def knn(D, class_var, k, categ): #assuming D is encoded and that all cols \neq class are being used
    preds = []

    D_ = encode_df(D, categ, class_var)


    for i in range(D.shape[0]):
        dists = []
        ite = 0
        print(i)
        
        curr = D_.iloc[i].to_numpy()
        matrix = D_.drop(i).to_numpy()

        for obs in matrix:
            dists.append(euclid(curr, obs))
            #dists.append(manhattan(curr, obs))
            #dists.append(cosine(curr, obs))

            ite += 1
       
        dists = np.array(dists)
        dists = np.argsort(dists)[:k] #returns index of k largest elements
        dists[dists >= i] += 1 #have to reindex since drop value at index i
        pred = D[class_var].iloc[dists].mode()[0]#prediction
        preds.append(pred)

    return preds

def mann(n1, n2): #convert to numpy arrays 
    return abs(n1 - n2)

def euclid(n1, n2):
    return np.linalg.norm(n1 - n2) 

def manhattan(n1, n2):
    return np.linalg.norm(n1 - n2, ord=1) 

def cosine(n1, n2):
    return np.dot(n1,n2) / (np.linalg.norm(n1)*np.linalg.norm(n2))

def encode_df(D, categ, class_var): #convert categ to numeric and normalize numeric!!!!!!!!!!!
    if len(categ) != 0:
        categ.remove(class_var) #if categorical after this is empty --> only numerical (no dummification needed)
    D_wo_class = D.drop(class_var, axis = 1)
    D_dum = pd.get_dummies(D_wo_class, columns = categ)
    D_dum = D_dum * 1 #makes sure that true and false are converted into 1 and 0
#df = df.apply(pd.to_numeric, errors='coerce')
    return D_dum

def output_stuff(D, preds, correct, class_var):
  output = []
  y_pred = pd.Series(preds)
  y_actu = D[class_var]
  not_correct = len(preds) - correct
  accu_rate = correct / len(preds)
  df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
  output.append("Total number of records classified:"  + str(len(preds))) #total number of records classified
  output.append("Total number of records correctly classified:" + str(correct))
  output.append("Total number of records incorrectly classified:" + str(not_correct))
  output.append("Overall Accuracy:" + str(accu_rate))
  output.append("Overall Error Rate:" + str(1 - accu_rate))
  output.append(df_confusion.shape)
  output.append(df_confusion.to_string())
  return output


def main():
    ret = classify.parser_check(sys.argv[1])
    D = ret[0] #need to decide how to encode THIS!!
    class_var = ret[1]
    categ = ret[2]
    
    k = int(sys.argv[2]) #temp

    preds = knn(D, class_var, k, categ)
    mask = [a == b for a, b in zip(preds, D[class_var])] #gross but should work
    count_correct = sum(mask)
    outs = output_stuff(D, preds, count_correct, class_var)
    for out in outs:
        sys.stdout.write(str(out) + "\n")


if __name__ == "__main__":
    main()
