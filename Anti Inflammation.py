# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 08:54:08 2023

@author: tejes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from minisom import MiniSom
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the actual file path
file_path = 'D:/projects/anti inflammatory/train10k.csv'

# Read CSV file into a DataFrame
df = pd.read_csv(file_path)

# Get column datatypes
column_datatypes = df.dtypes

# Count occurrences of each datatype
datatype_counts = column_datatypes.value_counts()

# Print datatype of each column
print("Column Datatypes:")
print(column_datatypes)

# Print count of each datatype
print("\nDatatype Counts:")
print(datatype_counts)

# Print number of non-null values in each column
print("\nNumber of Non-Null Values in Each Column:")
non_null_counts = df.count()
print(non_null_counts)

# Drop columns with no values
df = df.dropna(axis=1, how='all')

# Set the threshold for non-null values
threshold = 9602

# Drop columns with fewer than the specified non-null values
df = df.dropna(axis=1, thresh=threshold)


# Convert columns of data type int64 to float64
int_columns = df.select_dtypes(include='int64').columns
df[int_columns] = df[int_columns].astype('float64')

# Identify columns with object data type (categorical variables)
object_columns = df.select_dtypes(include='object').columns

# Create another DataFrame (assuming you already have it)
another_df = pd.DataFrame()

# Copy 'object' columns to the existing DataFrame
another_df[object_columns] = df[object_columns].copy()

# Display the DataFrame with 'object' columns copied
print("DataFrame with 'object' columns copied:")
print(another_df)

# Identify columns with object data type (categorical variables)
object_columns = another_df.select_dtypes(include='object').columns

# Convert 'object' columns to float64
another_df[object_columns] = another_df[object_columns].astype('float64')

# Display the DataFrame with 'object' columns converted to float64
print("DataFrame with 'object' columns converted to float64:")
print(another_df)

# Drop the column named 'name'
df = df.drop('name', axis=1)


# Use the last column as the target variable
y = df.iloc[:, -1]
y = pd.cut(y, bins=2, labels=False)
# Use all other columns as features (X)
X = df.iloc[:, :-1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a quadratic SVM model
svm_model = SVC(kernel='poly', degree=2)

# Train the model on the training set
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Train Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
print(f"Gaussian Naive Bayes Accuracy: {accuracy_gnb * 100:.2f}%")

# Train Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred_mnb = mnb.predict(X_test)
accuracy_mnb = accuracy_score(y_test, y_pred_mnb)
print(f"Multinomial Naive Bayes Accuracy: {accuracy_mnb * 100:.2f}%")

# Train Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
y_pred_bnb = bnb.predict(X_test)
accuracy_bnb = accuracy_score(y_test, y_pred_bnb)
print(f"Bernoulli Naive Bayes Accuracy: {accuracy_bnb * 100:.2f}%")


# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training set
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_classifier.predict(X_test)

# Evaluate the accuracy of the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")


# Use all columns as features (X)
X = df.values

# Check for constant features and remove them
constant_columns = np.where(np.max(X, axis=0) == np.min(X, axis=0))[0]
X = np.delete(X, constant_columns, axis=1)

# Normalize the data, handling possible division by zero
X_normalized = (X - np.min(X, axis=0)) / np.where((np.max(X, axis=0) - np.min(X, axis=0)) == 0, 1, (np.max(X, axis=0) - np.min(X, axis=0)))

# Specify the size of the SOM grid (e.g., 5x5)
som_grid_size = (5, 5)

# Create a SOM
som = MiniSom(som_grid_size[0], som_grid_size[1], X.shape[1], sigma=0.3, learning_rate=0.5)

# Initialize weights
som.random_weights_init(X_normalized)

# Train the SOM
num_epochs = 100
som.train_random(X_normalized, num_epochs)

# You can now use the trained SOM for various tasks, such as visualization or clustering
# For example, you can obtain the cluster assignments for each data point:
cluster_assignments = np.array([som.winner(x) for x in X_normalized])

# Print the cluster assignments
print("Cluster Assignments:")
print(cluster_assignments)

qe = som.quantization_error(X_normalized)
print(f"Quantization Error: {qe}")

te = som.topographic_error(X_normalized)
print(f"Topographic Error: {te}")

# Obtain the U-Matrix
u_matrix = som.distance_map()

# Plot the U-Matrix
plt.figure(figsize=(8, 8))
plt.pcolor(u_matrix, cmap='viridis', edgecolors='k', linewidths=0.1)
plt.colorbar()

# Add markers for each data point
for i, j in enumerate(cluster_assignments):
    plt.text(j[1] + 0.5, j[0] + 0.5, str(i + 1), color='red', ha='center', va='center', fontsize=8)

plt.title('SOM U-Matrix with Data Points')
plt.show()