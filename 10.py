#10 HIERARCHICAL CLUSTERING
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Load Data & Separate Labels
df = pd.read_csv(r"X:\ml lab\table\HIERARCHICAL_CLUSTERING.csv")
labels = df.pop("StudentID")

# 2. Cluster & Plot
mergings = linkage(df, method="complete")
dendrogram(mergings, labels=labels.values, leaf_rotation=90)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Student ID")
plt.ylabel("Distance")
plt.show()