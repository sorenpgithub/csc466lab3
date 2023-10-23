#CSC 466 Fall 2023 - Lab 3: Decision Trees, part 1
#Othilia Norell and Soren Paetau \\ onorell@calpoly.edu  / spaetau@calpoly.edu

#HOW TO RUN: 
#InduceC45  <TrainingSetFile.csv> [<restrictionsFile>] write

import pandas as pd
import numpy as np
import sys
import json

"""
Parsing the datafile and if provided also the restriction file
returns a dataframe with only the columns that should be used while inducing the decision tree
"""
def parser(filename,restfile): #utilize restfile, if no restfile assume None value
  #First three lines need to be parsed separately 
  with open(filename, 'r') as file:
    line1 = file.readline() #names of all columns
    line2 = file.readline() #information about the domain of each variable, −1: column is a rowId, 0: column is a numeric variable, n > 0 number of possible values of the variable
    line3 = file.readline() #class variable, unique for each dataset in this lab

  ##Classification algorithms must ignore rowId columns
  #If <restrictionsFile> is absent from the command line, the program shall use all non-ID attributes of the dataset to induce the decision tree
  line2_list = line2.split(',')
  line2_list = [l2.strip() for l2 in line2_list] #gets rid of white space and \n in col names
  
  cols = line1.split(',')
  cols = [col.strip() for col in cols] #gets rid of white space and \n in col names

  class_name = str(line3).strip()

  for i in range(len(line2_list)-1, -1, -1): #goes through the file backwards to make sure we are popping the right column
    if line2_list[i] == str(-1):
      cols.pop(i)

  # This (optional) file indicates which attributes of the dataset to use when inducing the decision tree. 
  #     A value of 1 means that the attribute in the corresponding position is to be used in the decision tree induction; 
  #     A value of 0 means that the attribute is to be omitted.
  if restfile is not None:
    with open(restfile, 'r') as rest_file:
        rest = rest_file.readline()
        rest_list = rest.split(',')
        rest_list = [r.strip() for r in rest_list] #gets rid of white space and \n in col names
    temp_cols = cols
    
    temp_cols.remove(class_name) #The size of the vector is equal to the number of columns in the dataset without the category variable
    
    for i in range(len(rest_list)-1, -1, -1): 
      if rest_list[i] == str(0):
            temp_cols.pop(i)
    temp_cols.append(class_name)
    intersection = [item for item in cols if item in temp_cols] #making sure only non-row ID columns and columns approved by restriction filed is being used to induce the decision tree
  else: #if restfile is not given
    intersection = cols 

  df_A = pd.read_csv(filename, usecols=intersection, names=intersection, skiprows=3) #creates a dataframe with the selected columns
 
  return (df_A, class_name)

#replace with string in 3rd row of CSV WILL BE GLOBAL
#list of categorical variables
"""
Selects the spliiting attribute from the dataframe that with a threshold has the highest information gain
Returns the attribute to split on
"""
def selectSplittingAttribute(A, D, threshold): #information gain
  p0 = enthropy(D) #\in (0,1) -sum
  gain = [0] * len(A)
  for i, A_i in enumerate(A): #i is index, A_i is string of col name
    p_i = enthropy_att(A_i, D)
    gain[i] = p0 - p_i #appending the info gain for each attribute to a list
  m = max(gain) #fidning the maximal info gain
  if m > threshold:
    max_ind = gain.index(m) #finding the list index of the maximal info gain
    return A[max_ind] #returns the attribute that had the maximal info gain
  else:
    return None

"""
Identifies the most frequent class label in the column specified by class_var
Returns both the label and its probability
"""
def find_freqlab(D): #assuming D is df
  values = D[class_var].value_counts(normalize = True)
  c = values.idxmax()
  pr = values[c]
  return (c,pr)

"""
Calculates the entropy of a dataset D based on a class variable class_var 
Entropy = -SUM(p*log2(p))
Returns 
"""
def enthropy(D):
  sum = 0
  bar = D.shape[0]
  for i in D[class_var].unique(): #each class label
    D_i = D[D[class_var] == i]
    foo = D_i.shape[0] #|D_i|
    pr = foo/bar
    sum += pr * np.log2(pr)
  return -sum

"""
Calculates the weighted entropy of a dataset D based on a particular attribute A_i
Weighted entropy = SUM(p*entropy(D_j))
Returns
"""
def enthropy_att(A_i, D):
  sum = 0
  bar = D.shape[0] #|D|
  for i in D[A_i].unique(): #for each value in domain of A_i
    D_j = D[D[A_i] == i]
    foo = D_j.shape[0] #|D_j|
    pr = foo/bar
    sum += pr * enthropy(D_j)
  return sum

"""
Helper function for edge cases
Constructs and returns a leaf node for a decision tree based on the most frequent class label
"""
def create_node(D):
  temp = find_freqlab(D) #should be whatever datatype c_i is
  r = {"leaf":{}}#create node with label of only class label STAR
  r["leaf"]["decision"] = temp[0]
  r["leaf"]["p"] = temp[1]
  return r

"""
Implements the C4.5 algorithm for building a decision tree 
from a dataset D based on a list of attributes A and a threshold value for information gain
Returns the (sub)tree T rooted at the current node
"""
def c45(D, A, threshold): #going to do pandas approach, assume D is df and A is list of col names
  #Edge case 1
  if D[class_var].nunique() == 1:
    #redundant to find mode if only one class label but bug proof!!
    T = create_node(D) #following exclusively psuedo code

  #Edge case 2
  elif not A:
    #redundant to find mode if only one class label but bug proof!!
    T = create_node(D)

  #"Normal" case
  else:
    A_g = selectSplittingAttribute(A, D, threshold) #string of column name
    if A_g is None:
      T = create_node(D)
    else:
     r = {"node": {"var":A_g, "edges":[]} } #dblcheck with psuedo code
     T = r
    #reference vs assignment are very delicate, triple check alg
    for v in D[A_g].unique(): #iterate over each unique value (Domain) of attribute (South, West..)
      D_v = D[D[A_g] == v] #dataframe with where attribute equals value
      if not D_v.empty: #true if D_v \neq \emptyset
        #test
        A_temp = A.copy()
        A_temp.remove(A_g)
        #print(A_temp)
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
        temp = find_freqlab(D) #should be whatever datatype c_i is
        r_v = {"leaf":{}} #create node with label of only class label STAR
        r_v["leaf"]["decision"] = temp[0]
        r_v["leaf"]["p"] = temp[1]
        #FINISH
                
      r["node"]["edges"].append(temp)
  return T


"""
Calling the C4.5 function
Returns the decision tree
"""
def get_tree(D, categ_vars, thresh):
  tree = c45(D, categ_vars, thresh)
  return tree


def initialize_global(path_file_in, rest_file_in, write_in = False):
  global path, rest_file, write, thresh, class_var
  path = path_file_in
  rest_file = rest_file_in
  write = write_in
   #determine best value
  ret = parser(path_file_in, rest_file_in)
  #print(ret, type(ret[0]))
  class_var = ret[1]


"""
Main function
"""
def main():
  #runs from command line
  write = False
  path_file = sys.argv[1]
  rest_file = None
  if len(sys.argv) >= 3:
    if sys.argv[2] != "None":
      rest_file = sys.argv[2]
    if len(sys.argv) >= 4 and sys.argv[3] == "write":
      write = True
  initialize_global(path_file, rest_file, write)
  
  thresh = 0.001 #determine best value
  categ_vars = list(D.columns)
  categ_vars.remove(class_var)
  tree = get_tree(D, categ_vars, thresh)
  
  out = {"dataset":path} #be careful if using path
  out.update(tree)
  
  json_obj =  json.dumps(out, indent = 4)
  sys.stdout.write(json_obj)

  if write:
    file_path = path.split(".")[0] + "_tree.json"
    # Open the file in write mode
    with open(file_path, 'w') as json_file:
        # Write the data to the file
        json.dump(tree, json_file)



if __name__ == "__main__":
   main()
