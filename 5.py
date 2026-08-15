#5 DECISION TREE ALGORITHM
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. Load Local Dataset (Update the path to your CSV file)
balance_data = pd.read_csv(r"X:\ml lab\table/DECISION_TREE_ALGORITHM.csv")

print("READ DATA\n")
print("Dataset Length: ", len(balance_data))
print("Dataset Shape: ", balance_data.shape)
print("\nDataset: \n", balance_data.head())

# 2. Separate Target (Y) and Features (X)
X = balance_data.values[:, 1:5]
Y = balance_data.values[:, 0]

# 3. Split Dataset into Training (70%) and Testing (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=100)

# 4. Decision Tree Classifier using Gini Index
clf_gini = DecisionTreeClassifier(
    criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

# Prediction using Gini
y_pred_gini = clf_gini.predict(X_test)

print("\nResults Using Gini Index:\n")
print("Predicted values:\n", y_pred_gini)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_gini))
print("\nAccuracy : ", accuracy_score(y_test, y_pred_gini) * 100)
print("\nReport :\n", classification_report(y_test, y_pred_gini))

# 5. Decision Tree Classifier using Entropy (Information Gain)
clf_entropy = DecisionTreeClassifier(
    criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

# Prediction using Entropy
y_pred_entropy = clf_entropy.predict(X_test)

print("\nResults Using Entropy:\n")
print("Predicted values:\n", y_pred_entropy)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_entropy))
print("\nAccuracy : ", accuracy_score(y_test, y_pred_entropy) * 100)
print("\nReport :\n", classification_report(y_test, y_pred_entropy))