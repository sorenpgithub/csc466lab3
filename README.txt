Othilia Norell and Soren Paetau \\ onorell@calpoly.edu  / spaetau@calpoly.edu

***HOW TO RUN CODE***
InduceC45.py: <TrainingSetFile.csv> [restrictionsFile.txt] [write]
TrainingSet and RestFile formatting defined in documentation.
if 'restrictionsFile' present then the program will only induce the decision tree based on the attributes corresponding to restrictionsFile's binary vector
if 'write' present then the program will write the json of the tree to disk under the name TrainingSetFile_tree.json

classify.py: <CSVfile.csv> <tree.json> [silent]
CSV file will be automatically determined whether it is training or not (has class variable in 3rd line) (not implemented)

evaluate.py:  <TrainingSetFile.csv> [restrictionsFile.txt] [n] [thresh]
Implements cross validation with n folds and optional threshold input.

randomForest.py: <TrainingSetFile.csv> <numAttributes> <numDataPoints> <numTrees>
Random Forest implication with training set file. Does not consider restrictions file nor will it check numAttributes is a valid value. Will automatically output a csv file named TrainingSetFile.results.csv with just predictions.

knn.py <TrainingSetFile.csv> [K]
Implements a basic K-nearest neighbor algorithms, returning the predictied outputs along with confidence matrix and associated metrics. No default values are set and outputs will be displayed in terminal.


***NOTE***
The strucutre of the functions and usage of global variables gives rise to concern as well. Overall this program needs to be cleaned up.

Further plans or additions:
-Test and finalize Nocross and allbutone functionality (UNTESTED)
-Add information gain RATIO along with just gain
-Further code commenting and clean up appearence
-Add input for threshold choice (currently hardcoded)
-Determine whether file is training (classify)

