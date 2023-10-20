import sys
import pandas as pd
import json

#classify.py CSVfile.csv tree.json silent

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

#print(df_A.shape[0])
def generate_preds(df_A, tree):
  pred = []
  correct = 0
  for index, row in df_A.iterrows(): #row is series object, val accessed like dict
    leaf = False
    curr_node = tree["node"] #{"var":123: "edges":[....]}
    #print(curr_node)

    while not leaf:
      #print("top loop", curr_node)
      A_i = curr_node["var"] 
      obs_val = row[A_i] #value of observation in variable, A_i = # of bedroom \implies obs_val = 3
      #print(A_i, obs_val)
      for edg in curr_node["edges"]: #list of edges | edg = {"edge":{"value"}}
        curr_edge = edg["edge"]
        #print("current edge", curr_edge)
        #print(curr_edge["value"], obs_val)
        if curr_edge["value"] == obs_val: 
          if "node" in curr_edge:
            curr_node = curr_edge["node"] #updating new node
            #print("changed", curr_node)
          else: #must be a leaf
            #print("in leaf statement")
            pred_val = curr_edge["leaf"]["decision"]
            
            if not silent:
              txt = "Row Id: {0} | Indicators: {1} | Pred: {2}".format(index, str(list(row)), pred_val)
              print(txt) #change from index to unique row_ids
            
            pred.append(pred_val)
            if is_training_csv and D[class_var].iloc[index] == pred_val:
                correct += 1
            leaf = True
          #print("found correct obs val")
          break
      #very very scary, always assumes there is edge with value
      #print("broken")
      #break
  return (pred, correct)

def output_stuff(preds, correct):
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
  output.append(df_confusion.to_string())
  return output

def initialize_global(D_in, class_var_in, training_in, silent_in = True): 
  global D, class_var, is_training_csv, silent
  D = D_in #Dataframe with all observations, currently assuming is training set
  class_var = class_var_in #string of colname of observed vals, could be None if not training
  is_training_csv = training_in #tbd on slack response
  silent = silent_in #if each classification prints
#should be able to directly call generate_preds after initializing

def main():
  silent_out = False
  with open(sys.argv[2]) as json_file: #turns json back into python dict
    tree = json.load(json_file)
  if len(sys.argv) == 4:
    silent_out = (sys.argv[3] == "silent")
     
  #initializing stuff
  #check if first element is leaf
  ret = parser_check(sys.argv[1])
  initialize_global(ret[0], ret[1], True, silent_out)
  df_A = D.drop(class_var, axis = 1) #drops class variable without c
  res = generate_preds(df_A, tree) #check if first node is leaf before calling!
  preds = res[0]
  correct = res[1]
  outs = output_stuff(preds, correct)
  for out in outs:
    sys.stdout.write(out + "\n")



if __name__ == "__main__":
   main()



  
