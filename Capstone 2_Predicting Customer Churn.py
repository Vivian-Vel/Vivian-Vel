#!/usr/bin/env python
# coding: utf-8

# In[118]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import statsmodels.api as sm
sns.set_style("darkgrid")
mpl.rcParams['figure.figsize'] = (20,5)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

get_ipython().run_line_magic('matplotlib', 'inline')


# Import Dataset

# In[119]:


df_tele =pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')


#                                  Exploratory Data Analysis

# In[120]:


df_tele.head(8)


# In[121]:


df_tele['Count'] = 1
ratios = df_tele.pivot_table(values="Count", index="Churn", aggfunc="sum")
ratios['Percentage'] = (ratios['Count']/ratios['Count'].sum())*100
ratios.head()


# In[122]:


sns.catplot(x='Churn', 
            data=df_tele,
            kind='count')
plt.title('Customer Churn Count',fontsize=15)
plt.show()


# The figure above shows us that 1869, 26.5% of the customers in this dataset churned. 5174, 73.5% of the customers stayed with the company.

# Lets do some more exploration, and look at some other variables observations

# In[123]:


colors = ['blue','orange']
df_tele.groupby(['Churn'])['MonthlyCharges'].mean().plot(kind='bar', width = 0.3,
                                                                stacked = True,
                                                                rot = 0, 
                                                                figsize = (10,6),
                                                                color = colors)
plt.title('Average MonthlyCharges within customers',fontsize=15)
plt.show()


# Convert target column into integers

# In[124]:


df_tele['Churn_Binary'] = df_tele['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)


# In[125]:


sns.barplot(x='gender',y='Churn_Binary',data=df_tele,ci=False)
plt.show()


# Gender has no effect on Churn. Both male and female are leaving at the same rate

# In[126]:


sns.barplot(x='SeniorCitizen',y='Churn_Binary',data=df_tele,ci=False)
plt.title('Churn within SeniorCitizen customers',fontsize=15)
plt.show()


# About43% of senior citezen customers are churning services

# In[127]:


df_tele.groupby(['Churn_Binary','SeniorCitizen'])['MonthlyCharges'].mean()


# Senior citizens that churned services had their monthly charges over the average $64. Senior citien customer that churned ervices had monthly charges over $80.7 

# In[128]:


sns.barplot(x='InternetService',y='Churn_Binary',data=df_tele,ci=False)
plt.title('Churn within internet customers',fontsize=15)
plt.show()


# 43% of the customers that churned services had fiber optic internet service

# In[129]:


Numeric_var= ['tenure', 'MonthlyCharges']
fig, ax = plt.subplots(1, 2, figsize=(28, 8))
df_tele[df_tele.Churn == 'No'][Numeric_var].hist(bins=20, color="blue", alpha=.4, ax=ax)
df_tele[df_tele.Churn == 'Yes'][Numeric_var].hist(bins=20, color="orange", alpha=.4, ax=ax)


# Over 1000 customers stayed with the company, however the monthly charges of those with a high tenure is about $20.Customers that churned in their first month of service had charges over $60 

# Feature Selection

# In[130]:


df_tele.columns


# From the columns above discriptive analysis, we can deduce that the following aren't relevant to our model development based on the subsequent reasons:
# -'CustomerID': this is a random identifier for a client and has no bearing on our model
# -'gender':  has no effect on Churn. Both male and female are leaving at the same rate
# -'Count': we added it simply to infer the chain ratio
# The rest of the columns are either numerical,ordinal or categorical, and will therefore play a part in our classification model. We can drop the ones we mentioned above.

# In[154]:


df_cleaned = df_tele.drop(['customerID','gender','Count'],axis=1,errors='ignore')


# In[155]:


df_cleaned.head()


# Data Preprocessing

# In[156]:


df_cleaned.describe(include='all')


# The longest tenure is 72 months or 6 years, average tenure is 32 months The maximum monthly charge is $118.75. The minimum monthly charge is $30.09. A customer at this company can expect to have a monthly charge of about $64.76. Assuming the charges are US dollars

# In[157]:


df_cleaned.info()


# Most of data type is object, "Yes or No". If we want to do further analysis, and predict churn, we need to be able to look at what variables contribute to churn. Forexample, if we use the linear or logistic regression models, we need to have numeric data. For the next steps, we are going to convert our data into floats or integers. 
# Our data will be converted to something like {'No': 0, 'Yes': 1}.

# Data cleaning

# Lets starte with the TotalCharges column

# In[158]:


sorted(df_tele['TotalCharges'].unique())
df_tele['TotalCharges_flot']=pd.to_numeric(df_tele['TotalCharges'], errors='coerce').astype(float)
df_tele['TotalCharges_flot'].dtype
df_tele['TotalCharges_flot']=df_tele['TotalCharges_flot'].fillna(df_tele['TotalCharges_flot'].mean())


# In[159]:


df_tele.isna().sum()


# Upon checking the dtypes, our new column 'TotalCharges_flot' is a float 

# MultipleLines column- has 'No phone service','Yes','No'. We can turn into a binary column by replacing 'No phone service' with 'No'

# In[160]:


df_cleaned['MultipleLines'].replace('No phone service', 'No').unique()


# In[226]:


df_cleaned['MultipleLines'] = df_cleaned['MultipleLines'].apply(lambda x: 1 if x == 'Yes' else 0


# OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies  columns- have 'No internet service','Yes','No'. We can turn them into a binary column by replacing 'No internet service' with 'No'

# In[248]:


df_cleaned([['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']].replace('No internet service', 'No')


# In[163]:


Binary_var= ['gender','Partner', 'Dependents','PhoneService','MultipleLines','OnlineSecurity','PaperlessBilling '
             'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies' ]


# In[164]:


ordinal_ver=[['Contract','InternetService']]


# In[165]:


nominal_var =['PaymentMethod']


# In[166]:


TERGET_COL =['Churn_Binary']


# In[167]:


internet_ordering=['No','DSL','Fiber optic']


# In[168]:


Contract_ordering=['Month-to-month', 'One year', 'Two year']


# In[249]:


df_cleaned.replace(to_replace=['No', 'Yes'], value=[0, 1],inplace=True)


# In[245]:


InternetService = pd.get_dummies(df_cleaned.InternetService).iloc[:,1:]


# In[246]:


InternetService.head()


# We can see from the dataframe produced above that our method worked. Now for the 'InternetService' column.

# In[229]:


df_cleaned.Contract.unique()


# 'Contract' has 3 unique values. If we simply attach numbers to correspond to the different cities, {'Month-to-month':0, 'One year':1, 'Two year':2}, the algorithm we design later will think that we assigned the numbers in order of importance. This is because of the ascension 0-1-2. This is something we don't want. The solution to this is once again one-hot encoding, but in a manner in which we only have 0s and 1s.

# In[176]:


Contract = pd.get_dummies(df_cleaned.Contract).iloc[:,1:]


# In[177]:


Contract.head()


# In[194]:


df_cleaned.PaymentMethod.unique()


# In[196]:


PaymentMethod = pd.get_dummies(df_cleaned.PaymentMethod).iloc[:,1:]


# In[197]:


PaymentMethod.head()


# As we can see from the dataframe above, we have 3 columns, Credit card (automatic),Electronic check, Mailed check.
# In fact, we can also apply this reasoning to the Contract dataframe we generated earlier. If we remove the first column, we can always infer its value as the opposite of the one that remains.

# In[200]:


df_cleaned.drop(['Contract','InternetService','PaymentMethod'],axis=1,inplace=True,errors='ignore')


# In[201]:


df_cleaned = pd.concat([df_cleaned,Contract,InternetService,PaymentMethod], axis=1)


# In[202]:


df_cleaned.head()


# In[250]:


pd.set_option("display.max_columns", None)

df_cleaned.head()


# Now that we've an entirely numerical dataset, we can move on to developing the model.

# In[255]:


df_cleaned['OnlineSecurity_Binary'] = df_cleaned['OnlineSecurity'].apply(lambda x: 1 if x == 'Yes' else 0)


# In[256]:


df_cleaned['OnlineBackup_Binary'] =df_cleaned['OnlineBackup'].apply(lambda x: 1 if x == 'Yes' else 0)


# In[257]:


df_cleaned['DeviceProtection_Binary'] = df_cleaned['DeviceProtection'].apply(lambda x: 1 if x == 'Yes' else 0)


# In[258]:


df_cleaned['TechSupport_Binary'] = df_cleaned['TechSupport'].apply(lambda x: 1 if x == 'Yes' else 0)


# In[259]:


df_cleaned['StreamingTV_Binary'] = df_cleaned['StreamingTV'].apply(lambda x: 1 if x == 'Yes' else 0)


# In[260]:


df_cleaned['StreamingMovies_Binary'] = df_cleaned['StreamingMovies'].apply(lambda x: 1 if x == 'Yes' else 0)


# In[291]:


df_cleaned2=df_cleaned.drop(['TotalCharges_flot','TotalCharges'],axis=1,errors='ignore')


# lets temporary drop TotalCharges columns.

# In[292]:


df_cleaned2.dtypes


# In[276]:


df_cleaned.dtypes


# In[293]:


df_cleaned2.columns


# In[296]:


x=df_cleaned2[['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
       'MultipleLines', 'PaperlessBilling','MonthlyCharges', 'One year', 'Two year', 'DSL', 'Fiber optic', 'No',
       'One year', 'Two year', 'DSL', 'Fiber optic', 'No',
       'Credit card (automatic)', 'Electronic check', 'Mailed check',
       'OnlineSecurity_Binary', 'OnlineBackup_Binary',
       'DeviceProtection_Binary', 'TechSupport_Binary', 'StreamingTV_Binary',
       'StreamingMovies_Binary', 'Partner_Binary']]


# In[297]:


y= df_cleaned2['Churn_Binary']


# In[298]:


x


# In[299]:


y


# Above, we defined 'y' as our label set, as it is the one we're going to be using for classification. It contains the decision markers of what constitutes a churned customer and what doesn't. It is the column from which the model is going to infer a prediction from.
# 
# As for 'X', it is our feature set. It contains the combination of features on which the decision in the labe' column is based.

# In[300]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[301]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
columns = x_train.columns


# In[302]:


x_test


# In[303]:


x_train


# In[304]:


from sklearn.linear_model import LogisticRegression


# In[305]:


model =LogisticRegression()


# In[306]:


model.fit(x_train, y_train)


# In[307]:


model.predict(x_test)


# In[308]:


model.score(x_train, y_train)


# In[309]:


model.predict_proba(x_test)


# In[ ]:


model.predict()


# Above, we split our data into a train set and a test set so that we can evaluate the model's performance when it is complete.
# 
# We'll be using scikit-learn's RandomForestClassifier for this project. Scikit-learn is a very reliable library for machine learning and the RFC is a model that can get the job done efficiently without much tweaking. In other words, it is recommended for fast deployment.

# Fitting and Predicting the Model

# In[310]:


classifier = RandomForestClassifier(n_estimators=200, random_state=1)  
classifier.fit(x_train, y_train)


# In[312]:


predictions = classifier.predict(x_test)


# In[313]:


print(accuracy_score(y_test, predictions))


# Our model has an accuracy score of 79%. Quite an acceptable figure. Now let's build a model using the entire dataset.

# In[314]:


final_classifier = RandomForestClassifier(n_estimators=200, random_state=0)  
final_classifier.fit(x, y)


# Before we make a prediction of an entity outside of our dataset using the final model, let's take a look at which features are contributing the most to our model.

# In[315]:


feat_importances = pd.Series(classifier.feature_importances_, index=x.columns).sort_values(ascending=False)
feat_importances.head()


# In[317]:


sns.set_style("white")
sns.set_palette("husl", 12)

plt.figure(figsize=(9,5))
ax = sns.barplot(x=feat_importances, y=feat_importances.index, orient="h")
ax.set_ylabel("")
ax.set_xlabel("Churn count", fontsize=12)
ax.set_title("Churn", fontweight="bold", fontsize=14)
ax.tick_params(labelsize=11)

sns.despine(top=True, right=True, left=True, bottom=True, ax=ax)
plt.show()


# From the barplot above, we can deduce that 'MonthlyCharges' is the largest contributor to our model.

# Conclusion

# Earlier, we noted that close to 26% of the clients in this dataset churned. Now we've a model that can predict churn with 79% accuracy. We can now direct resources towards developing a policy and strategy that will best retain the customers that are likely to leave the telephone company according to our model. This will save the company money because it is always cheaper to keep an existing client, than to search for a new one

# In[ ]:




