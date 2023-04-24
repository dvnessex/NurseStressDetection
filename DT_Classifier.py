
#Importing the necessary libraries
import numpy as np
import pandas as pd
import Data_Import,Data_Preprocess,Performance_Calculator
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
cv_1 = StratifiedKFold(shuffle=True,n_splits=10)
dt=DecisionTreeClassifier()

pipe_1 = Pipeline([('scaler',MinMaxScaler()),('dt', dt)])  # build pipeline
param_grid_1 = {
                 'dt__random_state':[0, 1, 2, 3, 4, 5, 10, 15,20,35,50],
                 'dt__criterion':['gini','entropy'],
                 }
search_1 = GridSearchCV(pipe_1, param_grid_1, n_jobs=-1)
scores_1 = cross_validate(search_1, input_train, output_train, scoring=['accuracy'], cv=cv_1, return_estimator=True)

#Model evaluation on Validation Data
Performance_Calculator.print_accuracy(scores_1)

y_predict = cross_val_predict(pipe_1, input_train, output_train, cv=cv_1)

Performance_Calculator.calculate_performance(output_train,y_predict)

#Model evaluation on Test Data
search_1.fit(input_train,output_train)
test_pred=search_1.best_estimator_.predict(input_test)

Performance_Calculator.calculate_performance(output_test,test_pred)

