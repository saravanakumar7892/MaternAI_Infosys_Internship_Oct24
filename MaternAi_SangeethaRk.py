#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv('pregnancy-risk-prediction-data-set.csv') 
df


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[10]:


df.isnull().sum()/df.shape[0]*100


# In[11]:


df.duplicated().sum()


# In[12]:


df.describe().style.background_gradient()


# In[13]:


#histogram for distribution
for i in df.select_dtypes("number").columns:
    sns.histplot(data=df,x=i)
    plt.show()


# In[14]:


for i in df.select_dtypes("number").columns:
    sns.boxplot(data=df,x=i)
    plt.show()


# In[15]:


for i in df.select_dtypes(exclude = 'object'):
    ax = sns.histplot(x = df[i], color = 'teal', bins = 15, kde = True)
    ax.set(xlabel = i, ylabel = 'Frequency')
    plt.show()


# In[16]:


df_clean = df.drop(columns=["Patient ID", "Name"])
df_clean


# In[ ]:





# In[17]:


label_encoder = LabelEncoder()
df_clean['Outcome'] = label_encoder.fit_transform(df_clean['Outcome'])


# In[18]:


x= df_clean.drop(columns=['Outcome'])
print(x)


# In[19]:


y = df_clean['Outcome']
print(y)


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)


# In[25]:


rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)


# In[26]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")


# In[27]:


print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[30]:


print("\n--- Predict Pregnancy Risk ---")
user_input = {}
for column in x.columns:
    user_value = input(f"Enter value for {column}: ")
    try:
        
        user_value = float(user_value)
    except ValueError:
       
        pass
    user_input[column] = user_value


# In[31]:


user_df = pd.DataFrame([user_input])


user_prediction = rf_model.predict(user_df)[0]

predicted_class = label_encoder.inverse_transform([user_prediction])[0]

print(f"\nPredicted Outcome: {predicted_class}")


# In[ ]:




