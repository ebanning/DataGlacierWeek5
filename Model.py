#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import data.
import sklearn
from sklearn import datasets
house = sklearn.datasets.fetch_california_housing(as_frame = True).frame
print(house.head())


# In[2]:


house = house.drop(['Latitude', 'Longitude'], axis = 1)
print(house.head())


# In[3]:


house['MedInc'] = house['MedInc']*10000
print(house.head())


# In[4]:


house['MedHouseVal'] = house['MedHouseVal'] * 100000
print(house.head())


# In[5]:


house = house.rename(columns = {'MedInc': 'Income', 'AveOccup': 'Occupancy', 'MedHouseVal': 'HousePrice'})
print(house.head())


# In[6]:


# Make model
import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array(house[['Income', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'Occupancy']])
y = np.array(house['HousePrice'])
model = LinearRegression().fit(X,y)


# In[ ]:


import pickle
filename = "C:/Users/estal/OneDrive/Desktop/DataGlacier/Week4/Model.sav"
pickle.dump(model, open(filename, 'wb'))


# In[7]:


import pandas as pd
features = np.array([[50000, 32, 7, 3, 300, 3]])
result = model.predict(features)
print(result)


# In[ ]:


features = [50000,32,7,3,300,3]
features = [np.array(features)]
result = model.predict(features)
print(result)


# In[ ]:


print(result[0])


# In[ ]:


result = round(result[0],2)
print(result)


# In[8]:


listf = [50000,32,7,3,300,3]
features = [np.array([float(x) for x in listf])]
model.predict(features)


# In[ ]:




