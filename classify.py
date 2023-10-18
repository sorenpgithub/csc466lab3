import sys
import pandas as pd
import json

#classify.py CSVfile.csv tree.json False
training = True
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


with open(sys.argv[2]) as json_file: #turns json back into python dict
    tree = json.load(json_file)

#initializing stuff
#check if first element is leaf
ret = parser_check(sys.argv[1])
class_var = ret[1]
D =  ret[0]
df_A = D.drop(class_var, axis = 1) #drops class variable without c


def generate_preds(df_A, tree):
  leaf = False
  pred = []
  correct = 0
  for index, row in df_A.iterrows(): #row is series object, val accessed like dict
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
            print([row])
            pred.append(pred_val)
            if training and D[class_var].iloc[index] == pred_val:
                correct += 1
            leaf = True
          break 
  return (pred, correct)

def output_stuff(preds, correct):
  output = []
  y_pred = pd.Series(preds)
  y_actu = D[class_var]
  not_correct = len(preds) - correct
  accu_rate = correct / len(preds)
  df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
  output.append("Total number of records classified:", len(preds)) #total number of records classified
  output.append("Total number of records correctly classified:", correct)
  output.append("Total number of records incorrectly classified:", not_correct)
  output.append("Overall Accuracy:", accu_rate)
  output.append("Overall Error Rate:", 1 - accu_rate)
  output.append(df_confusion)



print(df_A.columns())
res = generate_preds(df_A, tree)
preds = res[0]
correct = res[1]
outs = output_stuff(preds, correct)
for out in outs:
  sys.stdout.write(out)





  
