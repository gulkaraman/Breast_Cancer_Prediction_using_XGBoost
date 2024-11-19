# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the Dataset
# Reading the dataset from a CSV file
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
# columns = ["Sample code number", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]
# data = pd.read_csv(url, names=columns)
data = pd.read_csv("breast_cancer.csv")

# General information about the data
total_data = len(data)
print("Total number of data:", total_data)
print(data.info())

# Replace missing values represented by "?" with NaN
data.replace("?", np.nan, inplace=True)

# Fill missing values with the median and convert the column to a numeric format
data["Bare Nuclei"] = data["Bare Nuclei"].astype(float)  # Convert to float first if necessary
data["Bare Nuclei"] = data["Bare Nuclei"].fillna(data["Bare Nuclei"].median())
data["Bare Nuclei"] = data["Bare Nuclei"].astype(int)

# Map values in the "Class" column to 0 and 1 (0: benign, 1: malignant)
data["Class"] = data["Class"].map({2: 0, 4: 1})

# Count of healthy and diseased data
count_classes = pd.value_counts(data['Class'], sort=True)
count_classes.plot(kind='bar', rot=0)
plt.title("Count of Healthy and Diseased Data")
plt.xticks(range(2), ['Healthy', 'Diseased'])
plt.xlabel("Condition")
plt.ylabel("Frequency")
plt.show()

# Visualize the data
sns.countplot(x='Class', data=data)
plt.title("Count of Healthy and Diseased Data")
plt.xlabel("Condition")
plt.ylabel("Frequency")
plt.show()

# 2. Data Preprocessing
# Split the dataset into independent variables (X) and target variable (y)
X = data.drop(["Sample code number", "Class"], axis=1)
y = data["Class"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create the XGBoost Model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# 4. Train the Model
model.fit(X_train, y_train)

# 5. Evaluate the Model
y_pred = model.predict(X_test)

# 6. Calculate Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the first 10 rows of the dataset with columns
print("First 10 Rows of the Dataset:\n", data.head(10).to_string())

# 7. Visualizations
# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Display other performance metrics
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)

# Visualize feature distributions - Histogram
plt.figure(figsize=(12, 12))
for i, feature in enumerate(X.columns):
    plt.subplot(4, 5, i + 1)
    sns.histplot(data[feature], kde=True)
    plt.title(feature)
plt.tight_layout()
plt.show()

# Visualize feature distributions - Violin Plot
plt.figure(figsize=(12, 12))
for i, feature in enumerate(X.columns):
    plt.subplot(4, 5, i + 1)
    sns.violinplot(y=data[feature])
    plt.title(feature)
plt.tight_layout()
plt.show()

# Visualize the distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(x=data["Class"])
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()
