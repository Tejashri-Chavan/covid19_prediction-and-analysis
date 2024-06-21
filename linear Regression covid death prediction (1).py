#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snsi


# In[2]:


df = pd.read_csv('Covid_Death.csv')


# In[3]:


df.head()


# In[4]:


def date_to_binary(date):
    if date == '9999-99-99':
        return 0
    else:
        return 1


# In[5]:


df['DATE_DIED'] = df['DATE_DIED'].apply(lambda x: date_to_binary(x))


# In[6]:


df.head()


# In[7]:


def age_to_binary(age):
    if age >= 50 :
        return 1
    else:
        return 0


# In[8]:


df['AGE'] = df['AGE'].apply(lambda x: age_to_binary(x))


# In[9]:


df.head()


# In[10]:


replace_map1 = {97: 2, 98: 2, 99: 1}


# In[11]:


df['INTUBED'] = df['INTUBED'].replace(replace_map1)


# In[12]:


df.head(10)


# In[13]:


col1 = ['PNEUMONIA', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'ICU']
for i in col1:
    df[i] = df[i].replace(replace_map1)


# In[14]:


df.head()


# In[15]:


replace_map2 = {1: 0, 2: 1}


# In[16]:


df['SEX'] = df['SEX'].replace(replace_map2)


# In[17]:


df.head()


# In[18]:


replace_map3 = {1: 1, 2: 0}


# In[19]:


col2 = ['PATIENT_TYPE', 'INTUBED', 'PNEUMONIA', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'ICU']
for i in col2:
    df[i] = df[i].replace(replace_map3)


# In[20]:


df.head()


# In[21]:


df.rename(columns={"USMER": "LEVEL_TREATMENT", "PATIENT_TYPE": "RETURNED_HOME", "DATE_DIED": "DEATH", "INTUBED": "INTUBE", "HIPERTENSION": "HYPERTENSION", "CLASIFFICATION_FINAL": "COVID"}, inplace = True)


# In[22]:


replace_map4 = {1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0}


# In[23]:


df.head()


# In[24]:


df['COVID'] = df['COVID'].replace(replace_map4)


# In[25]:


df.head()


# In[26]:


X = df.drop(['COVID', 'DEATH'], axis = 1)
X.head()


# In[27]:


y_death = df['DEATH']
y_covid = df['COVID']


# In[28]:


y_death.head()
y_covid.head()


# # Death As Target

# In[39]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[40]:


X_train, X_test, y_death_train, y_death_test = train_test_split(X, y_death, test_size=0.19, random_state=42)


# In[41]:



# Train the Linear Regression model
linear_model_death = LinearRegression()
linear_model_death.fit(X_train, y_death_train)


# In[42]:


# Predict 'DEATH'
pred_death = linear_model_death.predict(X_test)


# In[43]:



# Evaluate the model
mse_death = mean_squared_error(y_death_test, pred_death)
r2_death = r2_score(y_death_test, pred_death)


# In[44]:


print(f"Mean Squared Error for 'DEATH': {mse_death}")
print(f"R^2 Score for 'DEATH': {r2_death}")


# In[45]:


# Convert predictions to binary
threshold = 0.5
binary_pred_death = (pred_death >= threshold).astype(int)


# In[46]:


accuracy_death = accuracy_score(y_death_test, binary_pred_death)
print(f"Accuracy is : {round(accuracy_death * 100, 2)} %")


# # COVID as Target

# In[47]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split


# In[48]:


X_train, X_test, y_covid_train, y_covid_test = train_test_split(X, y_covid, test_size=0.19, random_state=42)


# In[49]:


# Train the Linear Regression model
linear_model_covid = LinearRegression()
linear_model_covid.fit(X_train, y_covid_train)


# In[50]:


# Predict 'COVID'
pred_covid = linear_model_covid.predict(X_test)


# In[51]:


# Evaluate the model
mse_covid = mean_squared_error(y_covid_test, pred_covid)
r2_covid = r2_score(y_covid_test, pred_covid)


# In[52]:


print(f"Mean Squared Error for 'COVID': {mse_covid}")
print(f"R^2 Score for 'COVID': {r2_covid}")


# In[53]:


# Convert predictions to binary
threshold = 0.5
binary_pred_covid = (pred_covid >= threshold).astype(int)


# In[54]:


# Calculate accuracy
accuracy_covid = accuracy_score(y_covid_test, binary_pred_covid)
print(f"Accuracy is : {round(accuracy_covid * 100, 2)} %")


# In[ ]:




