#CSC 466 Fall 2023 - Lab 3: Decision Trees, part 1
#Othilia Norell and Soren Paetau \\ onorell@calpoly.edu  / spaetau@calpoly.edu

#HOW TO RUN: 
#classify.py CSVfile.csv tree.json silent

import sys
import pandas as pd
import json

"""
Parsing the datafile and if provided also the restriction file
returns a dataframe with only the columns that should be used while inducing the decision tree
"""
def parser_check(filename): #utilize restfile, if no restfile assume None value
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
    intersection = cols 

  df_A = pd.read_csv(filename, usecols=intersection, names=intersection, skiprows=3) #creates a dataframe with the selected columns
 
  return (df_A, class_name)

"""
Traverses the decision tree and makes predictions for a given dataset D based on the structure of the provided tree
Returns a tuple containing the list of predictions and the count of correct predictions
"""
def generate_preds(D, tree):
  df_A = D.drop(class_var, axis = 1)#makes new df, not inplace
  pred = []
  correct = 0
  
  if "leaf" in tree: #first element is a leaf
    dec = tree["leaf"]["decision"]
    pred = [dec]*D.shape[0]
    if is_training_csv:
      correct = (D[class_var] == pred).sum()
    return (pred, correct)
  
  for index, row in df_A.iterrows(): #row is series object, val accessed like dict
    leaf = False
    curr_node = tree["node"] #{"var":123: "edges":[....]}

    while not leaf:
      A_i = curr_node["var"] 
      obs_val = row[A_i] #value of observation in variable, A_i = # of bedroom \implies obs_val = 3

      for edg in curr_node["edges"]: #list of edges | edg = {"edge":{"value"}}
        curr_edge = edg["edge"]

        if curr_edge["value"] == obs_val: 
          if "node" in curr_edge:
            curr_node = curr_edge["node"] #updating new node

          else: #must be a leaf
            pred_val = curr_edge["leaf"]["decision"]
            
            if not silent: #print information
              txt = "Row Id: {0} | Indicators: {1} | Pred: {2}".format(index, str(list(row)), pred_val)
              print(txt) #change from index to unique row_ids
            
            pred.append(pred_val)
            
            #Checks if the prediction is right and adds a count if that is the case
            if is_training_csv and D[class_var].iloc[index] == pred_val:
                correct += 1
            leaf = True
          break
      #print("broken")
  return (pred, correct)

"""

"""
def output_stuff(D, preds, correct):
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

def initialize_global(class_var_in, training_in, silent_in = True): 
  global class_var, is_training_csv, silent
  #Dataframe with all observations, currently assuming is training set
  class_var = class_var_in #string of colname of observed vals, could be None if not training
  is_training_csv = training_in #tbd on slack response
  silent = silent_in #if each classification prints
#should be able to directly call generate_preds after initializing

"""
Main function
"""
def main():
  silent_out = False
  with open(sys.argv[2]) as json_file: #turns json back into python dict
    tree = json.load(json_file)
  if len(sys.argv) == 4:
    silent_out = (sys.argv[3] == "silent")
     
  #initializing stuff
  ret = parser_check(sys.argv[1])
  initialize_global(ret[1], True, silent_out) 
  #drops class variable without c
  D = ret[0]
  res = generate_preds(D, tree) #check if first node is leaf before calling!
  preds = res[0]
  correct = res[1]
  outs = output_stuff(D, preds, correct)
  for out in outs:
    sys.stdout.write(str(out) + "\n")



if __name__ == "__main__":
   main()



  
