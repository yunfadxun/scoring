#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import numpy as np


# In[2]:


df = pd.read_csv("Student_Marks.csv")
df


# In[3]:


df.info()


# In[4]:


#identifying outlier
for c in df:
    plt.figure()
    sns.boxplot(df[c],data=df)
    plt.title(f"boxplot for {c}")
    plt.show()


# In[5]:


#identifying correlation between each variable
sns.heatmap(df.corr(), annot=True, cmap="Greens")
    #apparently time to study does not really correlate number of courses
    #mark appears to be weakly correlated with number of courses as well
    #but for the sake of our model, we'll accompany both


# In[6]:


#defining x and y
x=df[["time_study","number_courses"]]
y=df["Marks"].values.reshape(-1,1)
print(x.shape,y.shape)


# In[7]:


#data splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, random_state=42, train_size= 0.75)


# In[8]:


#linear regression
from sklearn.linear_model import LinearRegression
basemodel= LinearRegression()
basemodel.fit(x_train,y_train)


# In[9]:


print(np.column_stack((y_train,basemodel.predict(x_train))))
    #it doesn't look that good tbh


# In[10]:


#predicting xtest
y_pred= basemodel.predict(x_test)
comparison_test= np.column_stack((y_test, y_pred))
print(comparison_test)


# In[11]:


#print r-score
from sklearn.metrics import r2_score
r2_train= r2_score(y_train, basemodel.predict(x_train))
r2_test= r2_score(y_test, y_pred)
print(f"your r2 score for train dataset is: ", r2_train)
print(f"your r2 test for test dataset is: ", r2_test)
    #oww that's reach


# In[12]:


#print linear regression equation
intercept= basemodel.intercept_
coeff= basemodel.coef_
print(f"your multiple linear regression equation for this model is: y=" + str(intercept[0]) + "+" + str(coeff[0][0]) + 
      "(time to study)" +  "+" + str(coeff[0][1]) + "(number of course you take)")


# In[ ]:


#user input
def user_entry():
    input1= float(input(f"please input your time spent to study in a week (floating answer is allowed): "))
    input2= int(input(f"please input your number of courses you take this semester (discrete only): "))
    
    features=[[input1, input2]]
    predicted_values= basemodel.predict(features)
    
    return predicted_values
predicted_values= user_entry()
print(f"your predicted final mark will be: ", predicted_values[0][0])


# In[13]:


predicted_values.shape


# In[ ]:




