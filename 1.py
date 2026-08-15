#1 DATA CLEANING
import pandas as pd
employee=pd.read_csv(r'X:\ml lab\table/DATA_CLEANING.CSV')
print(employee)

employee.drop_duplicates(subset="Name", keep='first', inplace=True)
print (employee)

missing=employee.isnull()
print(missing)


employee=employee.dropna(axis=0)
print(employee)

del employee['Sr.No']
employee['Project']=employee['Project'].str.replace('mobile','Mobile')
employee.columns=['EmployeeName','Address','Mobile','Domain','E-mailid']
print(employee)