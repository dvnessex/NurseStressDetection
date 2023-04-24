#Script to preprocess the given dataset of nurses


#This function is used to remove the features that is not helpful for model building and split the dataframe into the input and output arrays respectively
def preprocess_data(df_input):
  processed_dataframe=df_input.drop(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30'],axis=1)
  input_array = processed_dataframe.drop(['Stress'], axis=1).to_numpy()
  output_array = processed_dataframe.loc[:,'Stress'].to_numpy()
  print(input_array.shape)
  print(output_array.shape)
  return input_array,output_array