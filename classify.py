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
  if   
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
ret = parser_check(sys.argv[1])
D = ret[0]
ind_cols = [] #change


def prediction(row):
  pass

for
