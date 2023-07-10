from sklearn.preprocessing import LabelEncoder

items = ['Television', 'refrigerator', 'Microwave', 'Computer', 'Fan', 'Fan', 'Mixer', 'Mixer']

# Create a LabelEncoder object, and process label encoding with fit() and transform()
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
print("Encoded Values :", labels)
print("Encoded Class :", encoder.classes_)
print("Decoded Values :", encoder.inverse_transform([4, 5, 2, 0, 1, 1, 3, 3]))