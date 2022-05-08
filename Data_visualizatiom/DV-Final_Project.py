import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import linalg as LA
import numpy as np
import scipy.stats as st
import statsmodels.api as sm
import dash as dash
from dash import dcc
from dash import html
from dash.dependencies import Input,Output
import plotly.express as px
import numpy as np
from scipy.fft import fft
from dash.exceptions import PreventUpdate
import statistics
import warnings
warnings.filterwarnings('ignore')





#read the data
df=pd.read_csv('san-francisco-payroll_2011-2019.csv')


#Preproccessing

print(df.isnull().sum())
df=df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
status_percent_null=151501/len(df['Status'])
print(status_percent_null)

print(df.isnull().sum())
print(len(df))
df=df[df['Base Pay'] != 'Not Provided']
df=df[df['Benefits'] != 'Not Provided']
print(len(df))

job_title=['Transit Operator', 'Special Nurse', 'Registered Nurse', 'Firefighter', 'Custodian', 'Police Officer 3', 'Public Service Trainee', 'Recreation Leader', 'Public Svc Aide-Public Works', 'Patient Care Assistant']
df = df.loc[df['Job Title'].isin(job_title)]
#df['Job Title']=df.loc[df['Job Title'].isin([job_title])]


df['Overtime Pay'] = pd.to_numeric(df['Overtime Pay'])
df['Base Pay'] = pd.to_numeric(df['Base Pay'])
df['Other Pay'] = pd.to_numeric(df['Other Pay'])
df['Benefits'] = pd.to_numeric(df['Benefits'])
df['Total Pay'] = pd.to_numeric(df['Total Pay'])
df['Total Pay & Benefits'] = pd.to_numeric(df['Total Pay & Benefits'])
df['Year'] = pd.to_numeric(df['Year'])
print('this is the head of the dataset:\n',df.head())

#heatmap
corr=df.corr()
ax =sns.heatmap(corr,annot=True)
plt.show()

#outlier detection
df1=df.copy()
cols_names = ['Base Pay','Overtime Pay','Other Pay','Benefits','Total Pay','Total Pay & Benefits','Year']

for i in cols_names:
    q1_h, q2_h, q3_h = df1[i].quantile([0.25,0.5,0.75])

    IQR_h = q3_h - q1_h
    lower1 = q1_h - 1.5*IQR_h
    upper1 = q3_h + 1.5*IQR_h

    print(f'Q1 and Q3 of the {i} is {q1_h:.2f}  & {q3_h:.2f} ')
    print(f'IQR for the {i} is {IQR_h:.2f} ')
    print(f'Any {i} < {lower1:.2f}  and {i} > {upper1:.2f}  is an outlier')

    sns.boxplot(y=df1[i])
    plt.title(f'Boxplot of {i} before removing outliers')
    plt.show()


    df1 = df1[(df1[i] > lower1) & (df1[i] < upper1)]

    sns.boxplot(y=df1[i])
    plt.title(f'Boxplot of {i} after removing outliers')
    plt.show()

#PCA
X=df1[df1._get_numeric_data().columns.to_list()[:-1]]

H=np.matmul(X.T,X)
_,d,_=np.linalg.svd(H)
print(f'original data: singular values {d}')
print(f'original data: condition number {LA.cond(X)}')
sns.heatmap(corr,annot=True)
plt.title('co-eff co-rell between original features')
plt.show()

X=X.values
X=StandardScaler().fit_transform(X)
pca=PCA(n_components='mle',svd_solver='full')
pca.fit(X)
X_PCA=pca.transform(X)

print('Original Dim',X.shape)
print('Transform Data',X_PCA.shape)
print(f'explained variance ratio{pca.explained_variance_ratio_}')

#we can remove 2 features out of 6 features.
#we can keep 4 of the features to get the explained variance ratio more than 95.

plt.figure()
x=np.arange(1,len(np.cumsum(pca.explained_variance_ratio_))+1)
plt.xticks(x)
plt.plot(x,np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of componenets')
plt.ylabel('cumlative explained variance')
plt.title('cumlative explained variance vs number of componenets')
plt.show()

a,b=X_PCA.shape
print(a,b)

column=[]
for i in range(b):
    column.append(f'Principal Col{i}')
df_PCA=pd.DataFrame(data=X_PCA,columns=column)

total_df_redused=pd.DataFrame(df_PCA).corr()
sns.heatmap(total_df_redused,annot=True)
plt.title('heatmap for redused features')
plt.show()

print(df_PCA.head(5))

#########Normality test
kstest_Overtime_Pay = st.kstest(df['Overtime Pay'],'norm')
print(f"K-S test: statistics={kstest_Overtime_Pay[0]:.5f}, p-value={kstest_Overtime_Pay[1]:.5f}")
print(f"K-S test: Overtime-pay column looks {'Normal' if kstest_Overtime_Pay[1] > 0.01 else 'Non-Normal'}")

kstest_Base_Pay = st.kstest(df['Base Pay'],'norm')
print(f"K-S test: statistics={kstest_Base_Pay[0]:.5f}, p-value={kstest_Base_Pay[1]:.5f}")
print(f"K-S test: Base-pay column looks {'Normal' if kstest_Base_Pay[1] > 0.01 else 'Non-Normal'}")

kstest_Other_Pay = st.kstest(df['Other Pay'],'norm')
print(f"K-S test: statistics={kstest_Other_Pay[0]:.5f}, p-value={kstest_Other_Pay[1]:.5f}")
print(f"K-S test: other-pay column looks {'Normal' if kstest_Other_Pay[1] > 0.01 else 'Non-Normal'}")


kstest_Benefits = st.kstest(df['Benefits'],'norm')
print(f"K-S test: statistics={kstest_Benefits[0]:.5f}, p-value={kstest_Benefits[1]:.5f}")
print(f"K-S test: Benefit column looks {'Normal' if kstest_Benefits[1] > 0.01 else 'Non-Normal'}")

kstest_Total_Pay = st.kstest(df['Total Pay'],'norm')
print(f"K-S test: statistics={kstest_Total_Pay[0]:.5f}, p-value={kstest_Total_Pay[1]:.5f}")
print(f"K-S test: Total Pay column looks {'Normal' if kstest_Total_Pay[1] > 0.01 else 'Non-Normal'}")

kstest_Total_Pay_benefits = st.kstest(df['Total Pay & Benefits'],'norm')
print(f"K-S test: statistics={kstest_Total_Pay_benefits[0]:.5f}, p-value={kstest_Total_Pay_benefits[1]:.5f}")
print(f"K-S test: Total Pay & Benefits column looks {'Normal' if kstest_Total_Pay_benefits[1] > 0.01 else 'Non-Normal'}")

kstest_Year = st.kstest(df['Year'],'norm')
print(f"K-S test: statistics={kstest_Year[0]:.5f}, p-value={kstest_Year[1]:.5f}")
print(f"K-S test: Year column looks {'Normal' if kstest_Year[1] > 0.01 else 'Non-Normal'}")

#Normalize the overtime_pay
new_overtime_pay = st.norm.ppf(st.rankdata(df['Overtime Pay'])/(len(df['Overtime Pay']) + 1))
fig, axes = plt.subplots(2,2,figsize=(9,7))
sns.set_style('darkgrid')
sns.lineplot(x=np.linspace(1,54647,54647),y=df['Overtime Pay'],ax=axes[0,0]).set(xlabel= '# of samples',ylabel='Magnitude')
axes[0,0].set_title('Non-Gaussian data')

sns.set_style('darkgrid')
sns.lineplot(x=np.linspace(1,54647,54647),y=new_overtime_pay,ax=axes[0,1]).set(xlabel= '# of samples',ylabel='Magnitude')
axes[0,1].set_title('Transformed data (Gaussian)')

sns.set_style('darkgrid')
sns.histplot(x=df['Overtime Pay'],bins=100,ax=axes[1,0]).set(xlabel='Magnitude')
axes[1,0].set_title('Histogram of Non-Gaussian Data')

sns.set_style('darkgrid')
sns.histplot(x=new_overtime_pay,bins=100,ax=axes[1,1]).set(xlabel='Magnitude')
axes[1,1].set_title('Histogram of Transformed data (Gaussian)')
plt.tight_layout()
plt.show()

#Normalize the Base_pay
new_Base_pay = st.norm.ppf(st.rankdata(df['Base Pay'])/(len(df['Base Pay']) + 1))
fig, axes = plt.subplots(2,2,figsize=(9,7))
sns.set_style('darkgrid')
sns.lineplot(x=np.linspace(1,54647,54647),y=df['Base Pay'],ax=axes[0,0]).set(xlabel= '# of samples',ylabel='Magnitude')
axes[0,0].set_title('Non-Gaussian data')

sns.set_style('darkgrid')
sns.lineplot(x=np.linspace(1,54647,54647),y=new_Base_pay,ax=axes[0,1]).set(xlabel= '# of samples',ylabel='Magnitude')
axes[0,1].set_title('Transformed data (Gaussian)')

sns.set_style('darkgrid')
sns.histplot(x=df['Base Pay'],bins=100,ax=axes[1,0]).set(xlabel='Magnitude')
axes[1,0].set_title('Histogram of Non-Gaussian Data')

sns.set_style('darkgrid')
sns.histplot(x=new_Base_pay,bins=100,ax=axes[1,1]).set(xlabel='Magnitude')
axes[1,1].set_title('Histogram of Transformed data (Gaussian)')
plt.tight_layout()
plt.show()

#Normalize the Other Pay
new_other_pay = st.norm.ppf(st.rankdata(df['Other Pay'])/(len(df['Other Pay']) + 1))
fig, axes = plt.subplots(2,2,figsize=(9,7))
sns.set_style('darkgrid')
sns.lineplot(x=np.linspace(1,54647,54647),y=df['Other Pay'],ax=axes[0,0]).set(xlabel= '# of samples',ylabel='Magnitude')
axes[0,0].set_title('Non-Gaussian data')

sns.set_style('darkgrid')
sns.lineplot(x=np.linspace(1,54647,54647),y=new_other_pay,ax=axes[0,1]).set(xlabel= '# of samples',ylabel='Magnitude')
axes[0,1].set_title('Transformed data (Gaussian)')

sns.set_style('darkgrid')
sns.histplot(x=df['Base Pay'],bins=100,ax=axes[1,0]).set(xlabel='Magnitude')
axes[1,0].set_title('Histogram of Non-Gaussian Data')

sns.set_style('darkgrid')
sns.histplot(x=new_other_pay,bins=100,ax=axes[1,1]).set(xlabel='Magnitude')
axes[1,1].set_title('Histogram of Transformed data (Gaussian)')
plt.tight_layout()
plt.show()

#Normalize the Benefits
new_Benefits = st.norm.ppf(st.rankdata(df['Benefits'])/(len(df['Benefits']) + 1))
fig, axes = plt.subplots(2,2,figsize=(9,7))
sns.set_style('darkgrid')
sns.lineplot(x=np.linspace(1,54647,54647),y=df['Benefits'],ax=axes[0,0]).set(xlabel= '# of samples',ylabel='Magnitude')
axes[0,0].set_title('Non-Gaussian data')

sns.set_style('darkgrid')
sns.lineplot(x=np.linspace(1,54647,54647),y=new_Benefits,ax=axes[0,1]).set(xlabel= '# of samples',ylabel='Magnitude')
axes[0,1].set_title('Transformed data (Gaussian)')

sns.set_style('darkgrid')
sns.histplot(x=df['Benefits'],bins=100,ax=axes[1,0]).set(xlabel='Magnitude')
axes[1,0].set_title('Histogram of Non-Gaussian Data')

sns.set_style('darkgrid')
sns.histplot(x=new_Benefits,bins=100,ax=axes[1,1]).set(xlabel='Magnitude')
axes[1,1].set_title('Histogram of Transformed data (Gaussian)')
plt.tight_layout()
plt.show()

#Normalize the Total Pay
new_Total_pay = st.norm.ppf(st.rankdata(df['Total Pay'])/(len(df['Total Pay']) + 1))
fig, axes = plt.subplots(2,2,figsize=(9,7))
sns.set_style('darkgrid')
sns.lineplot(x=np.linspace(1,54647,54647),y=df['Total Pay'],ax=axes[0,0]).set(xlabel= '# of samples',ylabel='Magnitude')
axes[0,0].set_title('Non-Gaussian data')

sns.set_style('darkgrid')
sns.lineplot(x=np.linspace(1,54647,54647),y=new_Total_pay,ax=axes[0,1]).set(xlabel= '# of samples',ylabel='Magnitude')
axes[0,1].set_title('Transformed data (Gaussian)')

sns.set_style('darkgrid')
sns.histplot(x=df['Benefits'],bins=100,ax=axes[1,0]).set(xlabel='Magnitude')
axes[1,0].set_title('Histogram of Non-Gaussian Data')

sns.set_style('darkgrid')
sns.histplot(x=new_Total_pay,bins=100,ax=axes[1,1]).set(xlabel='Magnitude')
axes[1,1].set_title('Histogram of Transformed data (Gaussian)')
plt.tight_layout()
plt.show()

#Normalize the Total Pay & Benefits
new_Total_pay_benefits = st.norm.ppf(st.rankdata(df['Total Pay & Benefits'])/(len(df['Total Pay & Benefits']) + 1))
fig, axes = plt.subplots(2,2,figsize=(9,7))
sns.set_style('darkgrid')
sns.lineplot(x=np.linspace(1,54647,54647),y=df['Total Pay & Benefits'],ax=axes[0,0]).set(xlabel= '# of samples',ylabel='Magnitude')
axes[0,0].set_title('Non-Gaussian data')

sns.set_style('darkgrid')
sns.lineplot(x=np.linspace(1,54647,54647),y=new_Total_pay_benefits,ax=axes[0,1]).set(xlabel= '# of samples',ylabel='Magnitude')
axes[0,1].set_title('Transformed data (Gaussian)')

sns.set_style('darkgrid')
sns.histplot(x=df['Total Pay & Benefits'],bins=100,ax=axes[1,0]).set(xlabel='Magnitude')
axes[1,0].set_title('Histogram of Non-Gaussian Data')

sns.set_style('darkgrid')
sns.histplot(x=new_Total_pay_benefits,bins=100,ax=axes[1,1]).set(xlabel='Magnitude')
axes[1,1].set_title('Histogram of Transformed data (Gaussian)')
plt.tight_layout()
plt.show()

#Normalize the Year
new_year = st.norm.ppf(st.rankdata(df['Year'])/(len(df['Year']) + 1))
fig, axes = plt.subplots(2,2,figsize=(9,7))
sns.set_style('darkgrid')
sns.lineplot(x=np.linspace(1,54647,54647),y=df['Year'],ax=axes[0,0]).set(xlabel= '# of samples',ylabel='Magnitude')
axes[0,0].set_title('Non-Gaussian data')

sns.set_style('darkgrid')
sns.lineplot(x=np.linspace(1,54647,54647),y=new_year,ax=axes[0,1]).set(xlabel= '# of samples',ylabel='Magnitude')
axes[0,1].set_title('Transformed data (Gaussian)')

sns.set_style('darkgrid')
sns.histplot(x=df['Year'],bins=100,ax=axes[1,0]).set(xlabel='Magnitude')
axes[1,0].set_title('Histogram of Non-Gaussian Data')

sns.set_style('darkgrid')
sns.histplot(x=new_year,bins=100,ax=axes[1,1]).set(xlabel='Magnitude')
axes[1,1].set_title('Histogram of Transformed data (Gaussian)')
plt.tight_layout()
plt.show()

#Statistic
print('the median of the base pay is:',statistics.median(df['Base Pay']))
print('the mean of the base pay is:',statistics.mean(df['Base Pay']))

print('the median of the Overtime Pay is:',statistics.median(df['Overtime Pay']))
print('the mean of the Overtime Pay is:',statistics.mean(df['Overtime Pay']))

print('the median of the Other Pay is:',statistics.median(df['Other Pay']))
print('the mean of the Other Pay is:',statistics.mean(df['Other Pay']))

print('the median of the Benefits is:',statistics.median(df['Benefits']))
print('the mean of the Benefits is:',statistics.mean(df['Benefits']))

print('the median of the Total Pay is:',statistics.median(df['Total Pay']))
print('the mean of the Total Pay is:',statistics.mean(df['Total Pay']))

print('the median of the Total Pay & Benefits is:',statistics.median(df['Total Pay & Benefits']))
print('the mean of the Total Pay & Benefits is:',statistics.mean(df['Total Pay & Benefits']))


#Visualization
#line_plot
sns.lineplot(data='df',x=df['Year'],y=df['Total Pay & Benefits'])
plt.xlabel('year')
plt.ylabel('Total Pay and Benefits')
plt.title('line plot of Benefits and Salary changing during years')
plt.show()
#bar plot
ax = sns.barplot(x=df['Year'], y=df['Benefits'], hue="Status", data=df)
plt.xlabel('Year')
plt.ylabel('benefits')
plt.title('bar plot of Benefits in each year')
plt.show()
#count plot
ax = sns.countplot(x="Status", data=df)
plt.xlabel('Status')
plt.ylabel('number of status')
plt.title('count plot of full-time & Part-time employess')
plt.show()
#cat plot

sns.catplot(x = 'Job Title',y = 'Overtime Pay',data=df,
            kind ='bar',
            height = 8 , aspect= 1.5)
plt.title('cat plot of job title and overtime pay')
plt.show()

#pie plot for job title
df['Job Title'].value_counts().plot(kind='pie')
plt.title('pie plot for job title')
plt.show()

df_2014=df[df['Year'] == 2014]
df_2015=df[df['Year'] == 2015]
#df_2016=df[df['Year'] == 2016]
df_2017=df[df['Year'] == 2017]
df_2018=df[df['Year'] == 2018]
df_2019=df[df['Year'] == 2019]

s14=df_2014['Status'].value_counts().to_dict()
s15=df_2015['Status'].value_counts().to_dict()
#s16=df_2016['Status'].value_counts().to_dict()
s17=df_2017['Status'].value_counts().to_dict()
s18=df_2018['Status'].value_counts().to_dict()
s19=df_2019['Status'].value_counts().to_dict()

#sub plot and pie plot

fig=plt.figure(figsize=(10,8))
plt.subplot(3,2,1)
plt.pie(s14.values(), labels=s14.keys())
plt.title("pie chart of 2014 status")
plt.subplot(3,2,2)
plt.pie(s15.values(), labels=s15.keys())
plt.title("pie chart of 2015 status")

plt.subplot(3,2,3)
plt.pie(s17.values(), labels=s17.keys())
plt.title("pie chart of 2017 status")
plt.subplot(3,2,4)
plt.pie(s18.values(), labels=s18.keys())
plt.title("pie chart of 2018 status")
plt.subplot(3,2,5)
plt.pie(s19.values(), labels=s19.keys())
plt.title("pie chart of 2019 status")
plt.show()
#displot
sns.displot(x=df['Total Pay'],kde=False)
plt.title('the displot of Total pay')
plt.show()
#total pay is aroun 100k to 200K for most of the titles
#pairplot
sns.pairplot(df)
plt.title('The pairplot of the datset')
plt.show()
#heatmap
sns.heatmap(corr)
plt.title('The Heatmap of the datset')
plt.show()
#hist-plot
sns.histplot(data=df, x=df['Total Pay'], hue=df['Status'],element="step",
             stat="density", common_norm=False)
plt.title('the histplot of Total pay ')
plt.show()

#QQ plot
fig = sm.qqplot(df['Base Pay'],line ='45')
plt.title('QQ plot for Base Pay')
plt.show()

#Kernal density estimate
sns.kdeplot(data=df,
            x='Total Pay',
            y='Base Pay',
            hue='Status',
            fill=True)

plt.title('Status Kernel density plot')
plt.show()

#Scatter plot and regression line using sklearn
sns.regplot(x='Overtime Pay', y='Total Pay', data=df, logistic=True, ci=None)
plt.title('Scatter and Regression plot on overtime and Total pay')
plt.show()
# # #Multivariate Box plot
sns.boxplot(x='Year',y='Base Pay',data=df)
plt.title('Multivariate Box plot of each year based pay')
plt.show()
#violon plot
sns.set_theme(style="whitegrid")
Ax = sns.violinplot(x="Year", y="Total Pay & Benefits", data=df, palette="coolwarm")
plt.title('Violon plot of total-year and benefits base on the Year ')
plt.show()

