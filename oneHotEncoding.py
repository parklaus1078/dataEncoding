from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 10)
items = ['Television', 'refrigerator', 'Microwave', 'Computer', 'Fan', 'Fan', 'Mixer', 'Mixer']

# Transform items into 2-D ndarray
items = np.array(items).reshape(-1, 1)
print('items : \n', items)

# Applying One-Hot Encoding
oh_encoder = OneHotEncoder()
oh_encoder.fit(items)
oh_labels = oh_encoder.transform(items)

# The return of OneHotEncoder is Sparse array. Hence, it must be transformed into Dense array
print("One-Hot Encoded Data : \n", oh_labels.toarray())
print("One-Hot Encoded Data Dimension : \n", oh_labels.shape)

df = pd.DataFrame({'item': ['Television', 'refrigerator', 'Microwave', 'Computer', 'Fan', 'Fan', 'Mixer', 'Mixer']})
print('items in DataFrame : \n', df)
print(pd.get_dummies(df))