import sys
import pandas as pd
import json


training = False
silent = False #make argument in thing
def parser_check(filename): 
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


with open(sys.argv[2]) as json_file:
    tree = json.load(json_file)
#initializing stuff
#check if first element is leaf
ret = parser_check(sys.argv[1])
D = ret[0]
ind_cols = [] #change

def generate_preds(df_A, tree):
  leaf = False
  pred = []
  for index, row in df_A.iterrows():
    curr_node = tree["node"] #{"var":123: "edges":[....]}
    A_i = curr_node["var"] 
    obs_val = row[A_i] #value of observation in variable, A_i = # of bedroom \implies obs_val = 3
    while not leaf:
      for edg in curr_node["edges"]: #list of edges | edg = {"edge":{"value"}}
        curr_edge = edg["edge"]
        if curr_edge["value"] == obs_val: 
          if "node" in curr_edge:
            curr_node = curr_edge["node"] #scary since it changes list mid-iteration possibly bugs
          else: #must be a leaf
            pred_val = curr_edge["leaf"]["decision"]
            pred.append(pred_val)
            leaf = True
          break 
    return pred


  
