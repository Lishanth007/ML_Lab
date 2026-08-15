# 3 DATA REDUCTION (PCA)
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("X:\ml lab\table/DATA_REDUCTION.CSV")
print(df.head())

labler = df['Species']
x = df.drop(["Id", "Species"], axis=1)

variables = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
x = df.loc[:, variables].values
y = df.loc[:, ["Species"]].values
# transform data

x = StandardScaler().fit_transform(x)
x = pd.DataFrame(x)
print(x.head())

# fit the data using PCA
pca = PCA()
x_pca = pca.fit_transform(x)
x_pca = pd.DataFrame(x_pca)
print(x_pca.head())

# explained variance
explaind_variance = pca.explained_variance_ratio_
print(explaind_variance)

# plat data using PCA
plt.scatter(x_pca[1], x_pca[2])
plt.xlabel("PCA1")
plt.xlabel("PCA2")
plt.title("PCA PLOT")
plt.show()