
#Importing the necessary libraries
import numpy as np
import pandas as pd
import Data_Import,Data_Preprocess,Performance_Calculator
from sklearn.model_selection import train_test_split
from sklearn.tree import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_predict

#Data Import and Preprocessing
df_input=Data_Import.import_file()
input_array,output_array=Data_Preprocess.preprocess_data(df_input)

#Data Split
input_train, input_test, output_train, output_test = train_test_split(input_array, output_array, test_size=0.2, stratify=output_array)


#Cross Validation and Grid Search to select the best parameters
cv_2 = StratifiedKFold(shuffle=True,n_splits=10)
rf = RandomForestClassifier()
pipe_2 = Pipeline([('scaler',MinMaxScaler()),('rf', rf)])  # build pipeline
param_grid_2 = {
    'rf__max_depth': [3, 5, 10, 20]          # test different values for max_depth
}
search_2 = GridSearchCV(pipe_2, param_grid_2, n_jobs=-1)
scores_2 = cross_validate(search_2, input_train, output_train, scoring=['accuracy'], cv=cv_2, return_estimator=True)  

#Model evaluation on Validation Data
Performance_Calculator.print_accuracy(scores_2)

y_predict = cross_val_predict(pipe_2, input_train, output_train, cv=cv_2)

Performance_Calculator.calculate_performance(output_train,y_predict)

#Model evaluation on Test Data
search_2.fit(input_train,output_train)
test_pred=search_2.best_estimator_.predict(input_test)

Performance_Calculator.calculate_performance(output_test,test_pred)

