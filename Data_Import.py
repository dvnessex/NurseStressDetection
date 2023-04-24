
#Script to mount the google drive and import the unprocessed csv file from the drive

#Importing the necessary packages
from google.colab import drive
import os
import pandas as pd

#Defining the function to import, this function sets the path to the file, reads the file and returns the data frame
def import_file():
	
  drive.mount('/content/gdrive', force_remount=True)
  input_file=os.path.join('./CE888/Data/','combined_lagEDA.csv')
  input_file=os.path.join(os.path.join('gdrive', 'MyDrive', input_file))
  df_input=pd.read_csv(input_file)
  return df_input