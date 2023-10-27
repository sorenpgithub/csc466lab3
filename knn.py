#knn.py dataset.csv K 
import sys
import pandas as pd
import numpy as np 
import json
import classify



def knn(D, class_var, k): #assuming D is encoded and that all cols \neq class are being used
    preds = []
    for i in range(D.shape[0]):
        dists = []
        ite = 0

        curr = D.drop(class_var, axis = 1).iloc[i].to_numpy()
        matrix = D.drop(class_var, axis = 1).drop(i).to_numpy()

        for obs in matrix:
            dists.append(euclid(curr, obs))
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

def encode_df(D): #convert categ to numeric and normalize numeric!!!!!!!!!!!
    pass
#df = df.apply(pd.to_numeric, errors='coerce')


def main():
    ret = classify.parser_check(sys.argv[1])
    D = ret[0] #need to decide how to encode THIS!!
    class_var = ret[1]
    k = int(sys.argv[2]) #temp

    preds = knn(D, class_var, k)
    mask = [a == b for a, b in zip(preds, D[class_var])] #gross but should work
    count_correct = sum(mask)
    outs = classify.output_stuff(D, preds, count_correct, class_var)
    for out in outs:
        sys.stdout.write(str(out) + "\n")





if __name__ == "__main__":
    main()