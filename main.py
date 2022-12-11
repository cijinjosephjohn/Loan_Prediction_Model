import pandas as pd
import numpy as np
import plotly.express as px 
from matplotlib import pyplot as plt 

data = pd.read_csv("Default_Fin.csv")

print("first five data ")
print(data.head())

print("shape of the dataset")
print(data.shape)


print("check for null ")
print(data.isna().sum())

print("=====Data description=====")

print(data.describe())

data.insert(3,"Saving Rate",data["Bank Balance"]/data["Annual Salary"])

print(data.head())

table = data['Defaulted?'].value_counts().reset_index()
table.columns=["Status","Number"]
table["Status"] = table["Status"].map({1:"Defaulted",0:"Not Defaulted"})

print(table)



fig = px.pie(table,values="Number",names="Status",title="Default Status")


print(fig.show())

table = data["Employed"].value_counts().reset_index()
table.columns = ["Status","Number"]
table["Status"] = table["Status"].map({1:"Employed",0:"Unemployed"})

# print(table)

fig = px.pie(table,values="Number",names="Status",title="Employed Status")

fig.show()

table = data.copy()
table['Employed'] = table['Employed'].map({1 :'Employed', 0 :'Unemployed'})
table['Defaulted?'] = table['Defaulted?'].map({1 :'Defaulted', 0 :'Not defaulted'})

fig = px.sunburst(table, 
                  path=['Employed','Defaulted?'],
                  title='Default related with employment')
fig.show()


table = pd.crosstab(data['Employed'],data['Defaulted?'])
print(table)

from scipy.stats import chi2_contingency

chi2,p,dof,ex = chi2_contingency(table)

print(f"value for p : {p}")


fig = px.histogram(data,x="Bank Balance",color="Defaulted?",title="Bank Balance Distribution",marginal="box",hover_data = data.columns)

fig.show()

print(f"data of person's bal < = 10 : {(data['Bank Balance']<=10).sum()}")


fig =px.histogram(data,x="Annual Salary",color="Defaulted?",title="Annual Salary Distribution",marginal="box",hover_data = data.columns)

fig.show()

fig = px.histogram(data,x="Saving Rate",color="Defaulted?",marginal="box",hover_data=data.columns)
fig.show()


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(data.iloc[:,:-1],data.iloc[:,-1],test_size=0.3,stratify=data.iloc[:,-1],random_state=123)


print(f"shape of x_train : {x_train.shape},\nshape of x_test : {x_test.shape},\nshape of y_train : {y_train.shape},\nshape of y_test : {y_test.shape}")

import pickle

pickle_out = open("x_train","wb")
pickle.dump(x_train,pickle_out)

pickle_out = open("y_train","wb")
pickle.dump(y_train,pickle_out)

pickle_out = open("x_test","wb")
pickle.dump(x_test,pickle_out)

pickle_out = open("y_test","wb")
pickle.dump(y_test,pickle_out)

pickle_out.close()





