#Script to calculate the performance

#importing the necessary libraries
from sklearn.metrics import classification_report, f1_score, accuracy_score
import numpy as np

#This function prints the accuracy,f1-score and classification report by taking the expected and predicated array of labels as input
def calculate_performance(y_true, y_pred):
     

  acc = accuracy_score(y_true, y_pred)
  print('Accuracy Score: ', acc)

  f1score=f1_score(y_true, y_pred, average='macro')
  print('F1 Score(macro): ', f1score)


  print(classification_report(y_true, y_pred))


#This function prints the accuracy of each fold in cross validation
def print_accuracy(scores):
  print('Fold accuracy', scores['test_accuracy'])
  print('Average accuracy', np.mean(scores['test_accuracy']))