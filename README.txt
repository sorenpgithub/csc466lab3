Othilia Norell and Soren Paetau \\ onorell@calpoly.edu  / spaetau@calpoly.edu

***HOW TO RUN CODE***
InduceC45.py: <TrainingSetFile.csv> [restrictionsFile.txt] [write]
TrainingSet and RestFile formatting defined in documentation.
if 'restrictionsFile' present then the program will only induce the decision tree based on the attributes corresponding to restrictionsFile's binary vector
if 'write' present then the program will write the json of the tree to disk under the name TrainingSetFile_tree.json

classify.py: <CSVfile.csv> <tree.json> [silent]
CSV file will be automatically determined whether it is training or not (has class variable in 3rd line) (not implemented)

evaluate.py:  <TrainingSetFile.csv> [restrictionsFile.txt] [n]


***NOTE***
THIS PROGRAM IS NOT COMPLETE

The strucutre of the functions and usage of global variables gives rise to concern as well. Overall this program needs to be cleaned up.

Plans:
-Add f-measure and average accuracy!!
-Test and finalize Nocross and allbutone functionality (UNTESTED)
-Add information gain RATIO along with just gain
-Further code commenting and clean up appearence
-Optimize threshold choice (currently hardcoded)
-Analyze overfitting dillema
-Determine whether file is training (classify)
*

