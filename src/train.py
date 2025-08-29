#!/usr/bin/env python
# coding: utf-8

# # Step 1: Preparing the Data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.datasets import load_iris


# In[ ]:


iris = load_iris()


# In[ ]:


print("Iris data", iris.data)


# In[ ]:


iris["feature_names"]


# In[ ]:


iris["target_names"]


# In[ ]:


df = iris
df


# In[ ]:


df["target_names"]


# In[ ]:


X=iris.data
X


# In[ ]:


X.shape


# In[ ]:


y = iris.target
y


# In[ ]:





# In[ ]:


y.shape


# In[ ]:


print(iris.feature_names, iris.target_names)


# In[ ]:


X = pd.DataFrame(iris['data'], columns=iris["feature_names"])


# In[ ]:


x =  iris
x["target"]


# ## Split into training and test sets

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


y_train.shape


# In[ ]:


print("X_train shape:", X_train.shape)


# In[ ]:


X_train.head()


# # Scatter Plot

# In[ ]:


#sns.FacetGrid(iris, hue="target_names", height=6).map(plt.scatter, "Petal.Length", "Sepal.Width").add_legend()


# # Step 2: Choosing and Training a Model

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


model = DecisionTreeClassifier(random_state=42)


# ### Train (fit) the model
# 
# We train the decision tree using the training data:

# In[ ]:


model.fit(X_train, y_train)


# ## Step 3: Making Predictions

# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


print("Predictions:", y_pred[:5])


# In[ ]:


print("True labels:", y_test[:5])


# ## Step 4: Evaluating the Model
# After the model makes predictions on the test set, we need to evaluate how well it performed. For classification tasks like this, several evaluation metrics are commonly used:

# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy = accuracy_score(y_test, y_pred)


# In[ ]:


print("Accuracy:", accuracy)


# ### Improve data quality
# 
# Although sklearn.metric gave hundred percent accuracy, let us try another model to test their accuracy  

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
import joblib


# In[ ]:


model2 = KNeighborsClassifier(n_neighbors=5)


# In[ ]:


model2.fit(X_train, y_train)


# In[239]:


y_pred2 = model2.predict(X_test)


# In[240]:


y_pred2


# In[241]:


print("k-NN accuracy:", accuracy_score(y_test, y_pred2))


# In[242]:


print(classification_report(y_test, y_pred2))


# In[243]:


sns.heatmap(confusion_matrix(y_test, y_pred2),annot=True, cmap="Blues", fmt="d")
plt.savefig("confusion_matrix.png")
plt.show()


# In[247]:


dp = confusion_matrix(y_test, y_pred)
dp


# In[248]:


joblib.dump(model, "outputs/model.joblib")


# In[249]:


model = joblib.load("outputs/model.joblib")


# In[250]:


y_pred = model.predict(X_test)
print(y_pred)


# In[251]:


model = DecisionTreeClassifier(max_depth=3, random_state=42)


# In[252]:


model.fit(X_train, y_train)


# In[253]:


y_pred3 = model2.predict(X_test)


# In[254]:


y_pred3


# In[197]:


print("D-Tree accuracy:", accuracy_score(y_test, y_pred3))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




