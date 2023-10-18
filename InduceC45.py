import pandas as pd
import numpy as np
import sys
import json
#InduceC45  <TrainingSetFile.csv> [<restrictionsFile>]
print("hello world")
#Basic parser
def parser(filename):
  with open(filename, 'r') as file:
    line1 = file.readline()
    line2 = file.readline()
    line3 = file.readline() #3 lines are special, check doc

  cols = line1.split(',')
  cols = [col.strip() for col in cols] #gets rid of white space and \n in col names
  for i in range(len(line2)):
    if line2[i] == -1:
      cols.pop(i)

  class_name = str(line3).strip()
  df_A = pd.read_csv(filename, usecols=cols, names=cols, skiprows=3) #creates a dataframe

  return (df_A, class_name)

if len(sys.argv) == 1:
  print("Requires TrainingSet.csv")
  exit()

path = sys.argv[1]
print(path)
ret = parser(path)
D = ret[0]
class_var = ret[1]

categ_vars = list(D.columns)
print(class_var, categ_vars)

categ_vars.remove(class_var)
print(class_var, categ_vars)
#D.head()

#replace with string in 3rd row of CSV WILL BE GLOBAL
 #list of categorical variables
def selectSplittingAttribute(A, D, threshold): #information gain
  p0 = enthropy(D) #\in (0,1)
  gain = [0] * len(A)
  for i, A_i in enumerate(A): #i is index, A_i is string of col name
    p_i = enthropy_att(A_i, D)
    gain[i] = p0 - p_i #fancy maths stuff
  m = max(gain)
  if m > threshold:
    max_ind = gain.index(m)
    return A[max_ind]
  else:
    return None

def find_freqlab(D): #assuming D is df
  return D[class_var].mode()[0]

def enthropy(D):
  sum = 0
  bar = D.shape[0]
  for i in D[class_var].unique(): #each class label
    D_i = D[D[class_var] == i]
    foo = D_i.shape[0] #|D_i|
    pr = foo/bar
    sum += pr * np.log2(pr)
  return -sum

def enthropy_att(A_i, D):
  sum = 0
  bar = D.shape[0] #|D|
  for i in D[A_i].unique(): #for each value in domain of A_i
    D_j = D[D[A_i] == i]
    foo = D_j.shape[0] #|D_j|
    sum += (foo/bar) * enthropy(D_j)
  return sum

def c45(D, A, threshold): #going to do pandas approach, assume D is df and A is list of col names
  if D[class_var].nunique() == 1:
    c = find_freqlab(D) #should be whatever datatype c_i is
    r = {"leaf":{}}#create node with label of only class label STAR
    r["leaf"]["decision"] = c
    #redundant to find mode if only one class label but bug proof!!
    T = r #following exclusively psuedo code

  elif not A:
    c = find_freqlab(D) #should be whatever datatype c_i is
    r = {"leaf":{}}#create node with label of only class label STAR
    r["leaf"]["decision"] = c
    #redundant to find mode if only one class label but bug proof!!
    T = r

  else:
    A_g = selectSplittingAttribute(A, D, threshold) #string of column name
    if A_g is None:
      c = find_freqlab(D) #should be whatever datatype c_i is
      r = {"leaf":{}}#create node with label of only class label STAR
      r["leaf"]["decision"] = c
      #redundant to find mode if only one class label but bug proof!!
      T = r

    else:
     r = {"node": {"var":A_g, "edges":[]} } #dbl check with psuedo code
     T = r
    #reference vs assignment are very delicate, triple check alg


    for v in D[A_g].unique(): #terate over each unique value (Domain) of attribute (South, West..)
      D_v = D[D[A_g] == v] #dataframe with where attribute equals value
      if not D_v.empty: #true if D_v \neq \emptyset
        A_temp = A.copy()
        A_temp.remove(A_g)
        T_v = c45(D_v, A_temp, threshold)
        temp = {"edge":{"value":v}}
        #modify to contain edge value, look at lec06 example
        if "node" in T_v:
          temp["edge"]["node"] = T_v["node"]
        elif "leaf" in T_v:
          temp["edge"]["leaf"] = T_v["leaf"]
        else:
          print("something is broken")
        # r["node"]["edges"].append(temp)
      else: #ghost node
        r_v = {"leaf":{"decision":find_freqlab(D)}}

      r["node"]["edges"].append(temp)
  return T
#NEED TO ADD P TO LEAFS
#could use pandas?
thresh = 0.01 #determine best value
tree = c45(D, categ_vars, thresh) #
out = {"dataset":path} #be careful if using path
out.update(tree)
json_obj =  json.dumps(out, indent = 4)#should work
sys.stdout.write(json_obj) #might not work