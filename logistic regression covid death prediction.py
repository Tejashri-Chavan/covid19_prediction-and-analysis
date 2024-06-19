#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snsi


# In[ ]:





# In[ ]:





# In[3]:


df = pd.read_csv('Covid_Death.csv')


# In[4]:


df.head()


# In[5]:


def date_to_binary(date):
    if date == '9999-99-99':
        return 0
    else:
        return 1


# In[6]:


df['DATE_DIED'] = df['DATE_DIED'].apply(lambda x: date_to_binary(x))


# In[7]:


df.head()


# In[8]:


def age_to_binary(age):
    if age >= 50 :
        return 1
    else:
        return 0


# In[9]:


df['AGE'] = df['AGE'].apply(lambda x: age_to_binary(x))


# In[10]:


df.head()


# In[11]:


replace_map1 = {97: 2, 98: 2, 99: 1}


# In[12]:


df['INTUBED'] = df['INTUBED'].replace(replace_map1)


# In[13]:


df.head(10)


# In[14]:


col1 = ['PNEUMONIA', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'ICU']
for i in col1:
    df[i] = df[i].replace(replace_map1)


# In[15]:


df.head()


# In[16]:


replace_map2 = {1: 0, 2: 1}


# In[17]:


df['SEX'] = df['SEX'].replace(replace_map2)


# In[18]:


df.head()


# In[19]:


replace_map3 = {1: 1, 2: 0}


# In[20]:


col2 = ['PATIENT_TYPE', 'INTUBED', 'PNEUMONIA', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'ICU']
for i in col2:
    df[i] = df[i].replace(replace_map3)


# In[21]:


df.head()


# In[22]:


df.rename(columns={"USMER": "LEVEL_TREATMENT", "PATIENT_TYPE": "RETURNED_HOME", "DATE_DIED": "DEATH", "INTUBED": "INTUBE", "HIPERTENSION": "HYPERTENSION", "CLASIFFICATION_FINAL": "COVID"}, inplace = True)


# In[23]:


replace_map4 = {1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0}


# In[24]:


df.head()


# In[25]:


df['COVID'] = df['COVID'].replace(replace_map4)


# In[26]:


df.head()


# In[27]:


X = df.drop(['COVID', 'DEATH'], axis = 1)
X.head()


# In[28]:


y_death = df['DEATH']
y_covid = df['COVID']


# In[29]:


y_death.head()
y_covid.head()


# # Death As Target

# In[30]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_death_train, y_death_test = train_test_split(X, y_death, test_size = 0.19, random_state = 42)


# In[31]:


from sklearn.linear_model import LogisticRegression
model_death = LogisticRegression()
model_death.fit(X_train, y_death_train)


# In[ ]:





# In[32]:


pred_death = model_death.predict(X_test)


# In[33]:


from sklearn.metrics import accuracy_score
accuracy_death = accuracy_score(y_death_test, pred_death)


# In[34]:


print("Accuracy is :",round(accuracy_death * 100,2),"%")


# In[ ]:





# In[ ]:





# In[ ]:





# # COVID as Target

# In[35]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_covid_train, y_covid_test = train_test_split(X, y_covid, test_size = 0.19, random_state = 42)


# In[36]:


from sklearn.linear_model import LogisticRegression
model_covid = LogisticRegression()
model_covid.fit(X_train, y_covid_train)


# In[37]:


pred_covid = model_covid.predict(X_test)


# In[38]:


from sklearn.metrics import accuracy_score
accuracy_covid = accuracy_score(y_covid_test, pred_covid)


# In[39]:


print("Accuracy is :",round(accuracy_covid * 100,2),"%")


# In[ ]:





# In[ ]:





# In[ ]:




