import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.inspection import DecisionBoundaryDisplay

# 1. Load Data
X, y = datasets.load_iris(return_X_y=True)
X = X[:, :2]

# 2. Define & Fit Models with Titles
models = [
    ("SVC (Linear)", svm.SVC(kernel="linear")),
    ("LinearSVC", svm.LinearSVC(max_iter=10000)),
    ("SVC (RBF)", svm.SVC(kernel="rbf", gamma=0.7)),
    ("SVC (Poly)", svm.SVC(kernel="poly", degree=3, gamma="auto")),
]

# 3. Plot Decision Boundaries
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

for (title, clf), ax in zip(models, axes.ravel()):
    clf.fit(X, y)
    DecisionBoundaryDisplay.from_estimator(clf, X, response_method="predict", cmap="coolwarm", alpha=0.8, ax=ax)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=20, edgecolors="k")
    ax.set(title=title, xlabel="Sepal length", ylabel="Sepal width", xticks=[], yticks=[])

plt.tight_layout()
plt.show()