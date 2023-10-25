#knn.py dataset.csv K 
import sys
import pandas as pd
import json

def parser(path):
    pass


def knn(D, class_var, k): #assuming D is encoded
    flag = False
    clusters = init_centroids(D, k)
    D
    while not flag:
        for index,row in D.itterrows():
            for cluster in clusters:
                pass
    return clusters

def mann_dist(d1, d2):
    pass

def encode_df(D):
    pass

def init_centroids(D, k):
    #Pick k random points from D
    # 
    pass

def main():
    ret = parser(sys.argv[1])
    D = ret[0] #need to decide how to encode THIS!!
    class_var = ret[1]




if __name__ == "__main__":
    main()