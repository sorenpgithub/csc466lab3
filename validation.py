import sys
import pandas as pd
import json
import numpy as np

allbutone = False
nocross = False

#validation.py TrainingFile.csv restrictions.txt n
# which will take as input the training file, the optional restrictions file and an integer number n

def parser(filename):
    with open(filename, 'r') as file:
        line1 = file.readline()
        line2 = file.readline()
        line3 = file.readline() #3 lines are special, check doc

    cols = line1.split(',')
    cols = [col.strip() for col in cols] #gets rid of white space and \n in col names
    for i in range(len(line2)):
        if line2[i] == '-1':
          cols.pop(i)

  class_name = str(line3).strip()
  df_A = pd.read_csv(filename, usecols=cols, names=cols, skiprows=3) #creates a dataframe

def cross_val(df, k):
    shuffled_df = df.sample(frac=1)
    folds = np.array_split(shuffled_df, k) #k folds for cross validation

    for i in range(k): 
        test = folds[i]
        train = pd.concat([folds[j] for j in range(k) if j != i], ignore_index=True)

        #train the model
        
        accuracy =


if sys.argv[3] == -1:
    allbutone = True
elif sys.argv == 0:
    nocross = True
else:

    
    
