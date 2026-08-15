#2 DATA TRANSFORMATION
import numpy as np
import pandas as pd

df=pd.read_csv(r'X:\ml lab\table/DATA_TRANSFORMATION.CSV',header=0)
print(df)

df_categorial=df.select_dtypes(exclude=[np.number])
print(df_categorial)

x =df_categorial['Grade'].unique()
print(x)

y1=df_categorial['Grade'].value_counts()
print(y1)

y2 =df_categorial['Gender'].value_counts()
print(y2)

df.Grade.replace({"1st Class":1,"2nd Class":2,"3nd Class":3},inplace=True)
print(df)

df.Employed.replace({"yes":1,"no":0},inplace=True)
print(df)
df.head()