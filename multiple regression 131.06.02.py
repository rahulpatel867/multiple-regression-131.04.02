#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


cars = pd.read_csv("Cars.csv")
cars.tail()


# In[3]:


cars = pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# In[4]:


cars.info()


# In[5]:


cars.isna().diff()


# In[6]:


cars.isna().sum()


# In[7]:


fig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[8]:


fig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[9]:


fig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='MPG', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='MPG', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[10]:


fig,(ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[11]:


cars[cars.duplicated()]


# In[12]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[13]:


cars.corr()


# # OBSERVATIONS FROM CORRELATION AND COEFFICIENTS
>>> BETWEEN X AND Y, ALL THE X VARIABLES ARE SHOWING MODERATE TO HIGH CORRELATION STRENGHTS,HIGHEST BEING BETWEEN HP AND MP

>>> THEREFORE THIS DATASET QUALIFIES FOR BUILDING AMULTIPLE LINEAR REGRESSION MODEL TO PREDICT MPG

>>> AMONG X COLUMNS (x1,x2,x3,andx4),some very high correlation strengths are observed b/w SP VS HP, VOL VS WT

>>> THE CORRELATION AMONG X COLUMNS IS NOT DESIRABLE AS IS MIGHT LEAD TO MULTICOLLINEARITY PROBLEM
# In[14]:


model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[15]:


model1.summary()


# In[20]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["WT"]
df1.head()


# In[ ]:




