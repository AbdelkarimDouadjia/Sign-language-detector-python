# Import necessary libraries
import pickle  # For saving and loading data and models
from sklearn.ensemble import RandomForestClassifier  # RandomForestClassifier model
from sklearn.model_selection import train_test_split  # To split data into train and test sets
from sklearn.metrics import accuracy_score  # To evaluate model accuracy
import numpy as np  # For numerical operations

# Load the dataset from a pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))  # Load a dictionary containing 'data' and 'labels'

# Convert the data and labels into numpy arrays for processing
data = np.asarray(data_dict['data'])  # Feature data
labels = np.asarray(data_dict['labels'])  # Corresponding labels

# Split the data into training and testing sets
# 20% of the data will be used for testing, stratified by label distribution
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize a RandomForestClassifier model
model = RandomForestClassifier()

# Train the model on the training data
model.fit(x_train, y_train)

# Use the trained model to predict labels for the test data
y_predict = model.predict(x_test)

# Evaluate the accuracy of the model
score = accuracy_score(y_predict, y_test)  # Compare predictions with true labels

# Print the accuracy as a percentage
print('{}% of samples were classified correctly !'.format(score * 100))

# Save the trained model into a pickle file for later use
f = open('model.p', 'wb')  # Open file in write-binary mode
pickle.dump({'model': model}, f)  # Save the model into the file
f.close()  # Close the file
