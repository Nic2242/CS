"# CS" 
This repository contains 7 code files:

brama.py: This file is the main file running the duplicate detection procedure. It calls the different functions from the other files and creates the resulting F1, PQ, PC scores. 
plots.py: This file creates the plots shown in the paper. It reads the dataframes with the scores of the benchmark and the new method and ouputs a set of comparative plots
bral.py: This file contains the functions used to perform LSH in the main file.
bramsm.py: this file contains the necessary functions to run the MSM clustering in the main file
brami.py: this file contains the necessary functions to create signature representations of product pages in the main file. It contains the function to generate binary vectors and the function to generated signatures from those vectors, which are called seperately in the main file.
brat.py: this file contains the preprocessing and model word representation aspect of the method. The two important functions in this file give the value or title model words for a specific string. 
brat2.py: this file contains the benchmark processing and model word representation tools. 

To run the code for the benchmark, we must set all files that import brat to brat2. Then we run and store the results. For our model, all imports from brat2 must be changed to imports from brat. Then we run again and store the results. These two seperate results files are read in plots.py and concatenated to form the plots of the paper.
