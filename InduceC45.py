import pandas as pd
import numpy as np
import sys
import json
#InduceC45  <TrainingSetFile.csv> [<restrictionsFile>] write
#Basic parser
def parser(filename,restfile): #utilize restfile, if no restfile assume None value
  with open(filename, 'r') as file:
    line1 = file.readline()
    line2 = file.readline()
    line3 = file.readline() #3 lines are special, check doc

  line2_list = line2.split(',')
  line2_list = [l2.strip() for l2 in line2_list] #gets rid of white space and \n in col names
  
  cols = line1.split(',')
  cols = [col.strip() for col in cols] #gets rid of white space and \n in col names

  class_name = str(line3).strip()

  for i in range(len(line2_list)-1, -1, -1): #goes through the file backwards to make sure we are popping the right column
    if line2_list[i] == str(-1):
      cols.pop(i)

  if restfile is not None:
    # A restrictions file has been given
    with open(restfile, 'r') as rest_file:
        rest = rest_file.readline()
        rest_list = rest.split(',')
        rest_list = [r.strip() for r in rest_list] #gets rid of white space and \n in col names
    temp_cols = cols
    
    temp_cols.remove(class_name)
    for i in range(len(rest_list)-1, -1, -1): #the length of rest_list is one less than line2_list (since it should not contain the class attribute)
      if rest_list[i] == str(0):
            temp_cols.pop(i)
    temp_cols.append(class_name)
    intersection = [item for item in cols if item in temp_cols]
  
  else: 
    intersection = cols 

  df_A = pd.read_csv(filename, usecols=intersection, names=intersection, skiprows=3) #creates a dataframe
 
  return (df_A, class_name)

#replace with string in 3rd row of CSV WILL BE GLOBAL
 #list of categorical variables
def selectSplittingAttribute(A, D, threshold): #information gain
  entr = enthropy(D) #\in (0,1) -sum
  p0 = entr[0]
  prob = entr[1] 
  gain = [0] * len(A)
  for i, A_i in enumerate(A): #i is index, A_i is string of col name
    entr_att = enthropy_att(A_i, D)
    p_i = entr_att[0]
    prob_att = entr_att[1] #is this one needed for the probability?
    gain[i] = p0 - p_i #fancy maths stuff
  m = max(gain)
  if m > threshold:
    max_ind = gain.index(m)
    return A[max_ind]
  else:
    return None

def find_freqlab(D): #assuming D is df
  values = D[class_var].value_counts(normalize = True)
  c = values.idxmax()
  pr = values[c]
  return (c,pr)

def enthropy(D):
  sum = 0
  bar = D.shape[0]
  for i in D[class_var].unique(): #each class label
    D_i = D[D[class_var] == i]
    foo = D_i.shape[0] #|D_i|
    pr = foo/bar
    sum += pr * np.log2(pr)
  return (-sum,pr)

def enthropy_att(A_i, D):
  sum = 0
  bar = D.shape[0] #|D|
  for i in D[A_i].unique(): #for each value in domain of A_i
    D_j = D[D[A_i] == i]
    foo = D_j.shape[0] #|D_j|
    pr = foo/bar
    sum += pr * enthropy(D_j)[0]
  return (sum, pr)


def c45(D, A, threshold): #going to do pandas approach, assume D is df and A is list of col names
  if D[class_var].nunique() == 1:
    temp = find_freqlab(D) #should be whatever datatype c_i is
    r = {"leaf":{}}#create node with label of only class label STAR
    r["leaf"]["decision"] = temp[0]
    r["leaf"]["p"] = temp[1]
    #redundant to find mode if only one class label but bug proof!!
    T = r #following exclusively psuedo code

  elif not A:
    temp = find_freqlab(D) #should be whatever datatype c_i is
    r = {"leaf":{}}#create node with label of only class label STAR
    r["leaf"]["decision"] = temp[0]
    r["leaf"]["p"] = temp[1]
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
        r_v = {"leaf":{}}#create node with label of only class label STAR
        r_v["leaf"]["decision"] = temp[0]
        r_v["leaf"]["p"] = temp[1]
        #FINISH
                
      r["node"]["edges"].append(temp)
  return T
#NEED TO ADD P TO LEAFS
#could use pandas?


def get_tree(D, categ_vars, thresh):
  tree = c45(D, categ_vars, thresh)
  return tree

def initialize_global(path_file_in, rest_file_in, write_in = False):
  global path, rest_file, write, thresh, class_var, D
  path = path_file_in
  rest_file = rest_file_in
  write = write_in
   #determine best value
  ret = parser(path_file_in, rest_file_in)
  #print(ret, type(ret[0]))
  class_var = ret[1]
  D = ret[0]

  

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
  
  
  json_obj =  json.dumps(out, indent = 4)#should work
  sys.stdout.write(json_obj) #might not work

  if write:
    file_path = path.split(".")[0] + "_tree.json"
    # Open the file in write mode
    with open(file_path, 'w') as json_file:
        # Write the data to the file
        json.dump(tree, json_file)



if __name__ == "__main__":
   main()
