# CLASSIFICATION USING MULTILAYER PERCEPTRON
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# 1. Load Dataset
path = r"X:\ml lab\table\CLASSIFICATION_USING_MULTILAYER_PERCEPTRON.CSV"
bnotes = pd.read_csv(path)

# Data Exploration
print("Dataset Head:\n", bnotes.head())
print("\nUnique Classes:", bnotes["Class"].unique())
print("\nDataset Summary:\n", bnotes.describe(include="all"))

# 2. Features and Target Split
x = bnotes.drop("Class", axis=1)
y = bnotes["Class"]

# 3. Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)
print(f"\nTrain Shape: X={x_train.shape}, y={y_train.shape}")
print(f"Test Shape:  X={x_test.shape}, y={y_test.shape}")

# 4. Train Multilayer Perceptron (MLP)
mlp = MLPClassifier(
    hidden_layer_sizes=(3, 2), max_iter=500, activation="relu", random_state=42
)
mlp.fit(x_train, y_train)

# 5. Predictions & Evaluation
pred = mlp.predict(x_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))