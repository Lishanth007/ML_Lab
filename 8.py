import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. Load & Split Data
df = pd.read_csv(r"X:\ml lab\table\REGRESSION.csv")
X, y = df.iloc[:, :-1].values, df.iloc[:, 1].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

# 2. Train & Predict
model = LinearRegression().fit(X_tr, y_tr)
y_pred = model.predict(X_te)

# 3. Print Results
print("R² Score:", r2_score(y_te, y_pred))
print("Coefficient:", model.coef_[0], "Intercept:", model.intercept_)

# 4. Plot
plt.scatter(X_te, y_te, color="red")
plt.plot(X_te, y_pred, color="blue")
plt.title("Salary vs Experience")
plt.show()


