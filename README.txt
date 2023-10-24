Othilia Norell and Soren Paetau \\ onorell@calpoly.edu  / spaetau@calpoly.edu

***HOW TO RUN CODE***
InduceC45.py: <TrainingSetFile.csv> [restrictionsFile.txt] [write]
TrainingSet and RestFile formatting defined in documentation.
if 'restrictionsFile' present then the program will only induce the decision tree based on the attributes corresponding to restrictionsFile's binary vector
if 'write' present then the program will write the json of the tree to disk under the name TrainingSetFile_tree.json
classify.py: <CSVfile.csv> <tree.json> [silent]
CSV file will be automatically determined whether it is training or not (has class variable in 3rd line) (not implemented)
validation.py:  

***NOTE***
THIS PROGRAM IS NOT COMPLETE
evaluate has a major issue due to the structure of how variables are transfered to InduceC45. evaluate will not terminate due to 'Ghost nodes' not being properly implemented with the domain of A_g. 

Additionally, there is a slight issue with probabilities for presumably the same reason. Overfitting is major issue, such that classify is often returning perfect scores. Upon visual inspection tree apears to be of proper format but further testing required.

RestFiles are not fully properly implemted and the parser needs improvement on it's functionality and ability to handle edge cases. 

The strucutre of the functions and usage of global variables gives rise to concern as well. Overall this program needs to be cleaned up.

Plans:
-Add f-measure and average accuracy!!
-Add ability to handle continuous variables!!!
-Fix ghost node evaluation issue!!!
-Test and finalize Nocross and allbutone functionality (UNTESTED)
-Add information gain RATIO along with just gain
-Further code commenting and clean up appearence
-Optimize threshold choice (currently hardcoded)
-Analyze overfitting dillema
-Determine whether file is training (classify)
*

