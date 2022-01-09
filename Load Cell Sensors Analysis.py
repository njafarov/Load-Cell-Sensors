#!/usr/bin/env python
# coding: utf-8

# In[4]:


pwd


# In[5]:


import os


# In[6]:


os.chdir('/Users/njafarov/Downloads')


# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[8]:


df_64 = pd.read_excel('64.xlsx')


# In[9]:


df_64 = df_64.drop(columns=['Unit'], axis=0, inplace=False)


# In[ ]:





# In[10]:


collumns = ['Service_Life', 'Avg_Load', 'Load_More_Med', 'Average_Moisture', 'Moist_More_Med',
       'Ave_Vibration', 'Vib_More_Med', 'Ave_Solar_Exposure', 'Solar_More_Med', 'Loc_X',
       'LocX_More_Med', 'Loc_Y', 'LocY_More_Med', 'Censored', 'Infant', 'ExpireType']
df_64.columns = collumns


# In[ ]:





# In[11]:


df_64.describe()


# In[12]:


plt.figure(figsize=(14,8))

sns.pairplot(df_64)
plt.show()


# In[13]:


from sklearn.preprocessing import MinMaxScaler


# In[14]:


scaler = MinMaxScaler()


# In[16]:


x = df_64['Avg_Load']
y = df_64['Service_Life']


# In[17]:


x = scaler.fit_transform(x.array.reshape(-1,1))


# In[18]:


y = scaler.fit_transform(y.array.reshape(-1,1))


# In[19]:


sns.histplot(x)


# In[20]:


sns.histplot(y)


# In[21]:


plt.scatter(x,y)
plt.show()


# In[22]:


import statsmodels.api as sm


# In[23]:


x = sm.add_constant(x)


# In[24]:


model = sm.OLS(y,x)
results = model.fit()


# In[25]:


print(results.t_test([1,0]))


# In[26]:


df_64['Load_More_Med']


# In[95]:


from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder


# In[97]:


enc = OrdinalEncoder()
l_enc = LabelEncoder()

x_transformed = enc.fit_transform(enc_data)


# In[150]:


x_transformed[0]


# In[102]:


enc_data=df_64[['Load_More_Med', 'Moist_More_Med', 
       'Vib_More_Med', 'Solar_More_Med', 
       'LocX_More_Med', 'LocY_More_Med', 
       'ExpireType']]


# In[57]:


X = df_64.select_dtypes(include=[object])


# In[104]:


df_64[['Load_More_Med', 'Moist_More_Med', 'Vib_More_Med', 
       'Solar_More_Med', 'LocX_More_Med', 'LocY_More_Med', 
       'ExpireType']]= x_transformed


# In[ ]:





# In[105]:


df_64.head()


# In[39]:


sns.set_theme(font_scale=2)
plt.figure(figsize=(14,8))

sns.boxplot(x=df_64.Load_More_Med, y=df_64.Service_Life)
plt.title('Service life based on average load')
plt.savefig('box_plot_average_load.png')
plt.show()


# In[40]:


df_64.describe()


# In[106]:


X_1 = df_64.loc[df_64['Load_More_Med']==0, 'Service_Life']
X_2 = df_64.loc[df_64['Load_More_Med']==1, 'Service_Life']


# In[107]:


df_64.loc[df_64['Load_More_Med']==0, 'Service_Life']


# In[ ]:





# In[109]:


print(X_1.head())
print(X_2.head())


# In[110]:


import scipy.stats as stats


# In[111]:


f_value, p_value = stats.f_oneway(X_1, X_2)
print(f_value, p_value)


# In[112]:


import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[113]:


model = ols('Service_Life ~ Load_More_Med', data=df_64).fit()


# In[114]:


anova_tab = sm.stats.anova_lm(model, typ=2)


# In[116]:


print(anova_tab)



# In[117]:


plt.figure(figsize=(12,7))
sns.scatterplot(x=df_64['Avg_Load'], y=df_64['Service_Life'], color='r', s=92)
plt.title('Service life based on average load')
plt.savefig('Scatter_average_load.png')
plt.show()


# In[118]:


np.median(df_64['Avg_Load'])


# In[ ]:





# In[120]:


plt.figure(figsize=(12,7))
sns.scatterplot(x=df_64['Average_Moisture'], y=df_64['Service_Life'], color='r', s=92)
plt.title('Service life based on average moisture')
plt.savefig('Scatter_average_moist.png')
plt.show()


# In[121]:


sns.set_theme(font_scale=2)
plt.figure(figsize=(14,8))

sns.boxplot(x=df_64.Moist_More_Med, y=df_64.Service_Life)
plt.title('Service life based on average moisture')
plt.savefig('box_plot_average_moist.png')
plt.show()


# In[123]:


model = ols('Service_Life ~ Moist_More_Med', data=df_64).fit()
anova_tab = sm.stats.anova_lm(model, typ=2)
print(anova_tab)


#ANOVA Table: Service life based on level of moisture


# In[124]:


df_LMH = pd.read_excel('64_with_LMH.xlsx')


# In[125]:


df_LMH.head()


# In[126]:


from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# In[127]:


tukey = pairwise_tukeyhsd(endog=df_LMH['Service Life'],
                          groups=df_LMH['Load L M H'],
                          alpha=0.05)


# In[128]:


print(tukey)


# In[129]:


tukey = pairwise_tukeyhsd(endog=df_LMH['Service Life'],
                          groups=df_LMH['Moist L M H'],
                          alpha=0.05)
print(tukey)


# In[130]:


df_169 = pd.read_excel('169.xlsx')


# In[138]:


df_169.head()


# In[132]:


from sklearn.linear_model import LinearRegression


# In[133]:


lr = LinearRegression()


# In[135]:


x = df_169.drop(['Unit'], axis=1,inplace=True)


# In[139]:


x =df_169


# In[140]:


x.head()


# In[141]:


y = df_169['Service Life']


# In[142]:


X = sm.add_constant(x)


# In[256]:





# In[143]:


import statsmodels.api as sm


# In[144]:


ols = sm.OLS(y,X).fit()


# In[145]:


print(ols.summary())


# In[146]:


import matplotlib.pyplot as plt
plt.rc('figure', figsize=(12, 7))
#plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
plt.text(0.01, 0.05, str(ols.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig('output.png')


# In[147]:


print(X.shape)
print(y.shape)


# In[148]:


fig = plt.figure(figsize=(14,9))
fig = sm.graphics.plot_regress_exog(ols, 'Avg Load',fig=fig)


# In[149]:


fig = plt.figure(figsize=(14,9))
fig = sm.graphics.plot_regress_exog(ols, 'Average Moisture',fig=fig)


# In[297]:


df = pd.read_csv('_eba2c079135882131db3690701bc9c97_PASTAPURCHASE_EDITED.csv')


# In[326]:


df.head(40)


# In[ ]:





# In[313]:


df.groupby('AREA')['INCOME'].idxmax()


# In[336]:


df.loc[df.groupby('HHID')['PASTA'].idxmax()].sort_values(by='PASTA', ascending=False)


# In[343]:


df.groupby('AREA', as_index=False)['INCOME'].mean()


# In[338]:


(df.groupby(['AREA', 'PASTA'], as_index=True).mean()
            )


# In[345]:


df.corr()


# In[ ]:




