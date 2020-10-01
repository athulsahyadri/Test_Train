#!/usr/bin/env python
# coding: utf-8

# # IMPORTING RUDIMENTARY LIBRARIES

# In[1]:


import pandas as pd


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


import numpy as np


# In[4]:


import seaborn as sns


# In[5]:


from pandas_visual_analysis import VisualAnalysis  ##Pandas VisualAnalysis library


# # IMPORTING ASSISTIVE LIBRARIES

# In[38]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from statsmodels.regression.linear_model import OLS


# In[39]:


from sklearn.metrics import accuracy_score, roc_auc_score
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LogisticRegression


# In[40]:


from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score


# # Importing Data Sets

# In[9]:


df = pd.read_csv(r"C:\Users\Athul's PC-Asus TUF\Desktop\Class\DATA Mining\Assignment - Test Train\Train_Data(1).csv")


# In[10]:


df.shape


# In[11]:


df.describe()


# In[12]:


df.info()


# In[13]:


df.isnull().sum()


# In[14]:


df_Train =df


# # Data Analysis - Visualization and Exploration

# In[15]:


VisualAnalysis(df_Train)   ##Interactive Dashboard #Each Metric can be used to select the required 


# In[16]:


for indx in ['Income_of_Applicant', 'Income_of_Joint_Applicant', 'Loan_Amount_Requirement']:
 sns.boxplot(df_Train[indx].dropna(),orient='a')
 plt.title(indx)
 plt.show()


# # Dealing with missing value

# In[17]:


##from Out[10]

var_col =['Gender', 'Is_Married', 'No_of_Dependents', 'IS_Self_Employed', 'Loan_Amount_Term','Credit_History']

##Looping values

for indx in var_col:
    df_Train[indx]=df_Train[indx].fillna(df_Train[indx].mode()[0])


# In[18]:


## Method : Substitution of missing value -- Element : Median - int64

df_Train["Loan_Amount_Requirement"].fillna(df_Train["Loan_Amount_Requirement"].median(),inplace=True)


# #  data transformation 

# In[19]:


df_Train['Loan_Status'].replace('N', 0,inplace=True)
df_Train['Loan_Status'].replace('Y', 1,inplace=True)


# #  Creation of Dummy varibale 

# In[20]:


df_Train.head()


# In[21]:


df_Train=pd.get_dummies(df_Train)


# In[22]:


df_Train.head()


# In[ ]:





# # Splitting

# In[23]:


df_Train.info()


# In[24]:


##Splitting dataset - Train and Test - Ratio : 70:30
X=df_Train.drop("Loan_Status",1)
y=df_Train[["Loan_Status"]]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=1)


# In[25]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Imbalanced Dataset - Categorization and Resolution

# In[26]:


print(y_train["Loan_Status"].value_counts())
y_train["Loan_Status"].value_counts().plot.bar(title = 'Loan_Status')


# In[ ]:


#Status of dependent variable : Non-negligible imbalance - Requires resolution


# In[33]:


pip install --upgrade pip


# In[34]:


pip install imblearn


# In[27]:


from imblearn.over_sampling import SMOTE


# In[28]:


sm = SMOTE(random_state=1)
x_train1, y_train1 = sm.fit_sample(x_train, y_train)


# In[30]:


print(x_train1.shape)
print(y_train1.shape)


# # Transformation of Data

# In[54]:


## Method used : Dummy Training 




dumCol = ['Gender', 'Is_Married', 'No_of_Dependents', 'IS_Self_Employed', 'Level_of_Education',
              'Credit_History', 'Area_of_Property']
train_dummies = pd.get_dummies(df_Train[dumCol], drop_first = True)


# In[57]:


## Method used : Feature Normalization





numCol = ['Income_of_Applicant', 'Income_of_Joint_Applicant', 'Loan_Amount_Requirement', 'Loan_Amount_Term']
df_Train_num = (df_Train[numCol] - df_Train[numCol].mean()) / df_Train[numCol].std()


# In[60]:


##Method used:  Loan --> If approved : Value = 1 or Value = 0


Loan_St = df_Train.Loan_Status.apply(lambda x: 0 if x== 'N' else 1)
df_train = pd.concat ([df_Train_num, train_dummies, Loan_St], axis=1)


# In[61]:


df_train


# In[62]:


VisualAnalysis(df_Train_num)  ## Interactive Dashboard


# In[64]:


## Method Used : Correlation of Dataset - Grid Correlation


COR = df_Train.corr()
cg = sns.clustermap(COR, cmap ="YlGnBu", linewidths = 0.1); 
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0) 

cg


# In[65]:


mask = np.zeros_like(COR)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(COR, cmap='Spectral_r', mask=mask, square=True, vmin=-.4, vmax=.4)
plt.title('Correlation matrix')




###Correlation matrix. Warm colors (red) indicate a positive correlation, cool colors (blue) indicate a negative correlation.


# In[68]:


pip install pingouin  ## To extract the correlation value


# In[70]:


import pingouin as pg


# In[72]:



df_Train.rcorr(stars=False)


# # Running Models

# In[ ]:


#1)Logistic Regresssion Model


# In[44]:


#Implanting Regresssion model

lr_1=LogisticRegression(random_state=1)


# In[45]:


lr_1.fit(x_train1,y_train1)


# In[46]:


LogRegPred=lr_1.predict(x_test)


# In[52]:


acc = accuracy_score(LogRegPred,y_test)*100

acc


# #2)Random Forest Model

# In[54]:


frstMdl= RandomForestClassifier(random_state=1, max_depth=10,n_estimators=50)


# In[55]:


frstMdl.fit(x_train1,y_train1)


# In[57]:


prdFrstMdl=frstMdl.predict(x_test)


# In[58]:


acc = accuracy_score(prdFrstMdl,y_test)*100
acc


# # ACCURACY -- Logistic Model & Random Forest Model

# # Logistic Regression - Accuracy : 75.67 %

# # Random Forest - Accuracy : 78.37 %
