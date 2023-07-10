from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)

print("Average value of the features :", iris_df.mean())
print("Variance of the features :", iris_df.var())

# Creating Standard Scaler Object
stdScaler = StandardScaler()
# Transforming the dataset with StandardScaler
stdScaler.fit(iris_df)
# Calling fit() and transform()
iris_scaled = stdScaler.transform(iris_df)

# When transform(), dataset scaled is returned in ndarray. Hence, transforming it to DataFrame
iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
print("Average value of the features :")
print(iris_df_scaled.mean())
print("Variance of the features :")
print(iris_df_scaled.var())

# Creating Min Max Scalar
mmScaler = MinMaxScaler()
# Transforming the dataset into MinMaxScaler
mmScaler.fit(iris_df)
# Call fit() and transform()
iris_MinMaxScaled = mmScaler.transform(iris_df)

# The return of transform() is numpy ndarray. Hence, transform it into DataFrame
iris_df_MinMaxScaled = pd.DataFrame(data=iris_MinMaxScaled, columns=iris.feature_names)
print('Minimum value of the features :')
print(iris_df_MinMaxScaled.min())
print('Maximum value of the features :')
print(iris_df_MinMaxScaled.max())

# Points to be careful when using fit(), transform(), and fit_transform() with Scaler
# Create a train set that has sequence of integers from 0 to 10, but in array with dimension greater than 2.
train_array = np.arange(0, 11).reshape(-1, 1)
# Create a test set that has sequence of integers from 0 to 5, but in array with dimension greater than 2.
test_array = np.arange(0, 6).reshape(-1, 1)

# Create a MinMaxScaler object
newMmScaler = MinMaxScaler()
# When fit(), minimum and maximum value of train_array are set to 0 and 10.
newMmScaler.fit(train_array)
# Transform train_array data into 1/10 scale. Then, all the figures in the original data are multiplied by 1/10
train_scaled = newMmScaler.transform(train_array)

print('Original train_array data :', np.round(train_array.reshape(-1), 2))
print('Scaled train_array data :', np.round(train_scaled.reshape(-1), 2))

# When fit(), minimum and maximum value of test_array are set to 0 and 5.
newMmScaler.fit(test_array)
# Transform test_array data into 1/5 scale. Then all the figures in test_array are multiplied by 1/5
test_scaled = newMmScaler.transform(test_array)

print('Original test_array data :', np.round(test_array.reshape(-1), 2))
print('Scaled test_array data :', np.round(test_scaled.reshape(-1), 2))

# The Scale magnitudes of train data and test data are different in this case. This might cause problems
# Hence, When Scaling test data, fit() should not be called.
scaler = MinMaxScaler()
scaler.fit(train_array)
train_scaled = scaler.transform(train_array)
test_scaled  = scaler.transform(test_array)
print('Original train_array data :', np.round(train_array.reshape(-1), 2))
print('Scaled train_array data :', np.round(train_scaled.reshape(-1), 2))
print('Original test_array data :', np.round(test_array.reshape(-1), 2))
print('Scaled test_array data :', np.round(test_scaled.reshape(-1), 2))

test_scaled = scaler.fit_transform(test_array)
print('Original test_array data :', np.round(test_array.reshape(-1), 2))
print('Scaled test_array data :', np.round(test_scaled.reshape(-1), 2))