#!/usr/bin/env python
# coding: utf-8

# ### Welcome to the Airbnb Mini Practice Project
# 
# Throughout this unit on Python Data Analytics, you'be been introduced the the following powerful libraries:
# 
# <li> Matplotlib </li>
# <li> Seaborn </li>
# <li> Pandas </li> 
#     
# Each of these libraries will enhance your data analysis capabilities.
# 
# We've created this challenging exercise to reinforce your understanding of how these libraries work. 
# 
# Please note, there is a particular emphasis on the Pandas Library. This is the most critical Python library for data analytics. You'll see many similarities between Pandas and Pivot Tables!
#     
# <b> The most important thing you can do to build confidence with Python is to practice programming, all the time. This way you will build muscle memory. Don't simply copy the code you've written previously. Write it again and again so you store it in your memory. </b> 
# 
# <b> As this is a practice exercise, we've included a copy of what the outputs *should* look like for the majority of the questions to give you some guidance. </b>
# 
# <H3>  Time to get started! </H3>

# Import the airbnb_2.csv file.
# 
# Once you do this, you can start your analysis.
# 
# <b> Don't forget to import the libraries you need to read .csv files! </b> 
# 
# 

# ### Step 1: <span style="color:green">Import Libraries</span> 
# <b> Put your code in the box below. </b>
# 

# In[32]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Step 2: <span style="color:green">Import the Airbnb Dataset</span> 

# Now that you have the Pandas Libraries imported, it's time to import the Airbnb dataset.
# 
# <b> i) Import the Airbnb dataset.
# 
# ii) Use .info() function to better understand the variables inside your dataset.
# <p>    
# 
# <b> Put your code in the box below </b>

# In[33]:


df= pd.read_csv('airbnb_2.csv') 
df.head()


# In[34]:


print(df.info())


# ### Step 3: <span style="color:green">Exploring your data with Pandas</span> 
# 
# The rest of these questions will have you focus on using the following Pandas Skills:
# 
# <li> Subsetting a Pandas DataFrame using [] and boolean operators </li>
# <li> Summing up records with value_counts()</li>
# <li> Creating calculated fields </li>
# <li> Group By in Pandas </li> 
# <li> Creating Bar Plots with Matplotlib</li> 
# 
# 

# <b> i)  Please count how many Airbnb listings are in each of the 5 Neighbourhood Groups (Manhattan, Brooklyn, Queens, Bronx, Staten Island), then identify which Neighbourhood Groups have the greatest number of Airbnb listings. </b>
# <p>
#     <b> Put your code in the box below </b>

# In[35]:


print(df['neighbourhood_group'].value_counts())


# We want to focus our attention on the 3 most popular Neighbourhood Groups, by listing volume.
# 
# <b> ii) Calculate the percentage of Airbnb listings that each Neighbourhood Group contains. </b>
# 
# See this resource for more details <a href = https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html>. </a>
# 
# <b> Put your code in the box below. </b>

# In[36]:


most_popular_Neighbourhood_Groups = df.neighbourhood_group.value_counts(normalize=True).head(5)
print(most_popular_Neighbourhood_Groups)


# <b> iii) Create a new calculated field called Revenue and place this into the Airbnb DataFrame. This is to be calculated by using the Price Column x Number_Of_Reviews Columns </b>
# 
# <b> Put your code in the box below </b>

# In[37]:


df['Revenue'] = df['price']*df['number_of_reviews']
print(df)


# <b> iv) Create a Bar Plot that shows which Neighbourhood Group has the highest average revenues. In order to best
# calculate this, you'd want to consider how you can use the .groupby() syntax to assist you! </b>
# 
# If you're stuck, we recommend you go back to this <a href = https://learn.datacamp.com/courses/manipulating-dataframes-with-pandas> DataCamp link</a>. Specifically, Chapter 4 which covers how GROUP BY is used in Pandas.
# 
# <b> Put your code in the box below. </b>

# In[38]:


top_rev_neighbourhood_grp = df[['neighbourhood_group','Revenue']].groupby('neighbourhood_group').agg(['mean'])
top_rev_neighbourhood_grp.plot(kind='bar')
plt.show()


# <h3> <span style="color:green">Challenge Questions</span> </h3>

# <b> V) Filter the Airbnb DataFrame to include only the Neighbourhood Groups Manhattan, Brooklyn, and Queens. 
#     
# Then, identify the top 3 Revenue Generating Neighborhoods within each of the 3 Neighbourhood_Groups. This should give us 9 Overall Rows: 3 of the top generating neighbourhoods within each of the 3 Neighbourhood_Groups </b>
# 
# This is a tricky question that will *test* your group-by skills.
# 
# We recommend you consider the following:
# 
#     condition1 = someDataFrame['someColumn']=='someCondition'
#     condition2 = someDataFrame['someColumn']=='someCondition'
#     Da
#     Step One - Filter the taFrame using the Conditions
#     filtered_dataframe = someDataFrame[condition1 OR condition 2] 
#     #Hint: You might want to look up what the OR symbol in Python is represented as in operator form (i.e. AND (&) )
#     
#     Step Two - Group the Data by Neighbourhood_Group and Neighbourhood. Don't forget you're looking to SUM up the Revenues.
#     
#     The remaining steps we recommend you think very carefully about.
#     
#     You might want to make use of the .reset_index(inplace=True) function to help reset the indexes in 
#     your Grouped Up Dataframe...!
#     
#     
# <b> Put your code in the box below. </b>

# In[39]:


conditions1= df['neighbourhood_group'] =='Queens' 
conditions2= df['neighbourhood_group'] =='Manhattan' 
conditions3= df['neighbourhood_group'] =='Brooklyn'
filt_dataframe = df[conditions1|conditions2|conditions3]
sorte_dataframe = filt_dataframe.groupby(['neighbourhood_group','neighbourhood'])['Revenue'].sum().sort_values(ascending=False).reset_index()
top_3neigh_grps=sorte_dataframe.groupby('neighbourhood_group').head(3)
top_3neigh_grps


# In[50]:


#<b> VI) Filter the Airbnb Dataframe to include only the top 3 Neighbroos within each neighbourhood_group. 
    
#After doing this, identify the top average revenue-generating room-type for each of the nine neighbourhoods and plot this  in a Bar Chart.</b>

#This is a tricky question that will *test* your group-by skills. Think back to the previous question and how you approached this; you can approach this in a similar manner. 

#We recommend you consider the following:

conditions1= df['neighbourhood'] =='Williamsburg'
conditions2= df['neighbourhood'] =='Bedford-Stuyvesant'
conditions3= df['neighbourhood'] =='Harlem'
conditions4= df['neighbourhood'] =='Hell\'s Kitchen'
conditions5= df['neighbourhood'] =='Bushwick'
conditions6= df['neighbourhood'] =='Long Island City' 
conditions7= df['neighbourhood'] =='Flushing' 
conditions8= df['neighbourhood'] =='Astoria' 
conditions9= df['neighbourhood'] =='East Village' 

filte_top_3neighbro_dataframe = df[conditions1|conditions2|conditions3|conditions4|conditions5|conditions6|conditions7|conditions8|conditions9]
sorte_top_3neighbro_dataframe = filte_top_3neighbro_dataframe.groupby([ 'neighbourhood','room_type']).sum()['Revenue'].sort_values(ascending=False)
sorte_top_3neighbro_dataframe.groupby('neighbourhood').head(1).plot(kind='bar')



    #Step One - Filter the Dataframe using the Conditions
#filtered_dataframe = someDataFrame[condition1 OR condition 2] 
    #Hint: You might want to look up what the OR symbol in Python is represented as in operator form (i.e. AND (&) )
    
    #Step Two - Group the Data by Neighbourhood_Group and Neighbourhood. Don't forget you're looking to SUM up the Revenues.
    
    #The remaining steps we recommend you think very carefully about.
    
    #You might want to make use of the .reset_index(inplace=True) function to help reset the indexes in 
    #your Grouped Up Dataframe...!
    
    
# <b> Put your code in the box below. </b>  


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




