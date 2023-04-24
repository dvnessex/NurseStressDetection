# NurseStressDetection

## Overview:

This project here implements the design and test of a machine learning model that can predict the stress based on the signals received from various sensors wore by the nurses.


This repository contains the following subsections of code. All of them are in python source code.

Data_Import.py : This section of implements helps in mounting the google drive and reading the input dataset (in the form of csv).
Data_Preprocess.py : This section preprocesses the data and creates the input and output arrays.
Performance_Calculator.py: This calculates the performance metrices for folds in cross validation and also the overall scores
DT_Classifier.py and RF_Classifier.py: This code runs through the training, validating the model on test data. Here, the decision tree and random forest classifiers were used in the respective files


## Prerequisties:

The environment is setup to be run on the colab IDE by Google and hence the dataset and scripts files has to be imported and setup accordindly on the google drive for proper running of the code.


## Reference:

### Project:
S. Hosseini, R. Gottumukkala, S. Katragadda, R. T. Bhupatiraju, Z. Ashkar, C. W. Borst, and K. Cochran.
A multimodal sensor dataset for continuous stress detection of nurses in a hospital. Scientific Data

### Dataset: 
Hosseini, S. et al. A multi-modal sensor dataset for continuous stress detection of nurses in a hospital. Dryad https://doi.org/10.5061/dryad.5hqbzkh6f
