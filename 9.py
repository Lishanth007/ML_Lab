#9 K-MEANS CLUSTERING
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# 1. Load Data
X = pd.read_csv(r"X:\ml lab\table\K-MEANS_CLUSTERING.csv").iloc[:, [3, 4]].values

# 2. Elbow Method Plot
wcss = [KMeans(n_clusters=i, random_state=42).fit(X).inertia_ for i in range(1, 11)]
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Graph")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.show()

# 3. Fit K-Means & Plot Clusters
kmeans = KMeans(n_clusters=5, random_state=0)
y = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="rainbow", s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=150, c="black", marker="X")
plt.title("Customer Groups")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()