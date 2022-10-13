#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dataset = pd.read_csv("C:/Users/lenovo/Downloads/Social_Network_Ads.csv")


# In[3]:


dataset.head()


# In[4]:


x = dataset.drop('Purchased',1)
y = dataset['Purchased']


# In[5]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test=train_test_split(x ,y, test_size=0.25 , random_state = 0)


# In[13]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

    


# In[15]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier (criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)


# In[16]:


y_pred = classifier.predict(x_test)


# In[17]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[ ]:





# In[ ]:




