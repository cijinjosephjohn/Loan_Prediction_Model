import pandas as pd
import numpy as np
import plotly.express as px 
from matplotlib import pyplot as plt 

data = pd.read_csv("Default_Fin.csv")

# print("first five data ")
# print(data.head())

# print("shape of the dataset")
# print(data.shape)


# print("check for null ")
# print(data.isna().sum())

# print("=====Data description=====")

# print(data.describe())

data.insert(3,"Saving Rate ",data["Bank Balance"]/data["Annual Salary"])

# print(data.head())

# table = data['Defaulted?'].value_counts().reset_index()
# table.columns=["Status","Number"]
# table["Status"] = table["Status"].map({1:"Defaulted",0:"Not Defaulted"})

# print(table)



# fig = px.pie(table,values="Number",names="Status",title="Default Status")


# print(fig.show())

# table = data["Employed"].value_counts().reset_index()
# table.columns = ["Status","Number"]
# table["Status"] = table["Status"].map({1:"Employed",0:"Unemployed"})

# # print(table)

# fig = px.pie(table,values="Number",names="Status",title="Employed Status")

# fig.show()

# table = data.copy()
# table['Employed'] = table['Employed'].map({1 :'Employed', 0 :'Unemployed'})
# table['Defaulted?'] = table['Defaulted?'].map({1 :'Defaulted', 0 :'Not defaulted'})

# fig = px.sunburst(table, 
#                   path=['Employed','Defaulted?'],
#                   title='Default related with employment')
# fig.show()


table = pd.crosstab(data['Employed'],data['Defaulted?'])
print(table)