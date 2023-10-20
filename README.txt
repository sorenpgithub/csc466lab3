Othilia Norell and Soren Paetau \\ onorell@calpoly.edu  / spaetau@calpoly.edu

InduceC45.py: <TrainingSetFile.csv> [restrictionsFile.txt] [write]
TrainingSet and RestFile formatting defined in documentation.
if 'write' present then the program will write the json of the tree to disk under the name TrainingSetFile_tree.json
classify.py: <CSVfile.csv> <tree.json> [silent]
CSV file will be automatically determined whether it is training or not (has class variable in 3rd line)
validation.py:

Note:
Currently the way the program is setup could be improved: Consistent use of global variables along with initilzaing functions create some redundancy and spots for issues to occour. 
