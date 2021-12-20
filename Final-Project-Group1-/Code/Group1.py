import requests
import pydot
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from io import StringIO
import requests
from scipy.stats import zscore
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sci
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
warnings.filterwarnings("ignore")
import requests

import sys
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from io import StringIO
from sklearn.metrics import roc_curve, auc
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Link to the Kaggle data set & name of zip file
login_url = 'https://www.kaggle.com/divyansh22/online-gaming-anxiety-data?select=GamingStudy_data.csv'

# Kaggle Username and Password
kaggle_info = {'UserName': "iaasish123@gwmail.gwu.edu", 'Password': "Password@123"}
r = requests.post(login_url, data=kaggle_info, stream=True)
df = pd.read_csv('GamingStudy_data.csv',encoding='cp1252')

print(df)
#HeatMap
# #convert the variables to numeric
df["Age"] = pd.to_numeric(df["Age"])
df["Hours"] = pd.to_numeric(df["Hours"])
df["streams"] = pd.to_numeric(df["streams"])
df["Hours"]=pd.to_numeric(df['Hours'])
df["GAD_T"]=pd.to_numeric(df['GAD_T'])

#Asish
#heat maps
# gx = df.pivot("Age","Hours", "GAD_T")
# print(gx.info())
#Heatmap1 for general features
col1 = ['Timestamp','Hours','streams','Age','Narcissism','GAD_T']
cor = df[col1].corr()
ax = sns.heatmap(cor, annot=True, annot_kws={'fontsize':9})
plt.show()

#Heatmap2 for GAD Disorder
col2 = ['GAD1','GAD2','GAD3','GAD4','GAD5','GAD6','GAD7','GADE','GAD_T']
cor = df[col2].corr()
ax1 = sns.heatmap(cor, annot=True, annot_kws={'fontsize':10})
plt.show()

#Heatmap3 for SWL Disorder:
col3 = ['SWL1','SWL2','SWL3','SWL4','SWL5','SWL_T','GAD_T']
cor = df[col3].corr()
ax2 = sns.heatmap(cor, annot=True, annot_kws={'fontsize':10})
plt.show()

#Heatmap4 for SPIN Disorder:
col4 = ['SPIN1','SPIN2','SPIN3','SPIN4','SPIN5','SPIN6','SPIN7','SPIN8','SPIN9','SPIN10','SPIN11','SPIN12','SPIN13','SPIN14','SPIN15','SPIN16','SPIN17','SPIN_T','GAD_T']
cor = df[col4].corr()
ax2 = sns.heatmap(cor, annot=True, annot_kws={'fontsize':5})
plt.show()
# drop the coloumns which we are not working with and have the high corelation with GAD_T

df = df.drop(columns=['Narcissism','streams','SPIN1','SPIN2','SPIN3','SPIN4','SPIN5','SPIN6','SPIN7','SPIN8','SPIN9','SPIN10','SPIN11','SPIN12','SPIN13','SPIN14','SPIN15','SPIN16','SPIN17','Timestamp','accept','League','Birthplace','Reference','Birthplace_ISO3','highestleague','SWL1','SWL2','SWL3','SWL4','SWL5','earnings','whyplay','Birthplace_ISO3','Residence_ISO3'])
df.drop(df[(df['Playstyle']!='Singleplayer') & (df['Playstyle']!='Multiplayer - online - with strangers') & (df['Playstyle']!='Multiplayer - online - with online acquaintances or teammates') & (df['Playstyle']!='Multiplayer - online - with real life friends') & (df['Playstyle']!='Multiplayer - offline (people in the same room)') & (df['Playstyle']!='all of the above')].index,axis=0,inplace=True)

#convert the variables to numeric
df["Age"] = pd.to_numeric(df["Age"])
print(df["Age"].dtype)

df["Hours"] = pd.to_numeric(df["Hours"])
#df["streams"] = pd.to_numeric(df["streams"])
#df.dropna(subset=['SPIN_T'])

####Asish
df = df.dropna()

box_plot_data=df[['Hours']]
plt.boxplot(box_plot_data)
plt.title("Hours")
plt.show()
box_plot_data=df[['Age']]
plt.boxplot(box_plot_data)
plt.title("Age")
plt.show()
box_plot_data=df[['GAD_T']]
plt.boxplot(box_plot_data)
plt.title("GAD_T")
plt.show()

df = df[(-3 < zscore(df['Hours'])) & (zscore(df['Hours']) < 3)]
df = df[(-3 < zscore(df['Age'])) & (zscore(df['Age']) < 3)]
df = df[(-3 < zscore(df['GAD_T'])) & (zscore(df['GAD_T']) < 3)]
df = df[(-3 < zscore(df['SWL_T'])) & (zscore(df['SWL_T']) < 3)]
print(df)

#df=df.drop(df.loc[df['GAD_T']<1].sample(frac=0.20).index)
#EDA
#Age
df.plot.scatter(x='Hours',y='Age', marker='o',figsize=(7,5))
plt.show()
#MB# platform vs hour
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
sns.catplot(x="Platform", y="Hours", data=df)
plt.show()

#
sns.set(style="darkgrid")
sns.catplot(x="Gender", y="Hours", data=df)
plt.show()

#MB
#df['GAD_T'] = pd.cut(x=df['GAD_T'], bins=4, labels=['minimal anxiety', 'mild anxiety', 'moderate anxiety','severe anxiety'])
gad_new = []
for i in df['GAD_T']:
    if i<=4:
        gad_new.append('mild')
    elif ((i>=5)&(i<=9)):
        gad_new.append('moderate')
    elif (i>=10):
        #&(i<=14)):
        gad_new.append('moderately severe')
    #elif i>=15:
    # gad_new.append('severe')
df['GAD_T'] = gad_new
sns.histplot(x="GAD_T", y="Age", data=df)
plt.show()

#SWL

# sns.set(style="darkgrid")
# sns.catplot(x="SWL_T", y="GAD_T", data=df)
# plt.show()
#Game
game = df['Game'].value_counts()
game.plot(kind='bar',figsize=(10,8))
plt.title('Game')
plt.show()
#PlayStyle
game = df['Playstyle'].value_counts()
game.plot(kind='bar',figsize=(10,8))
plt.title('Playstyle')
plt.show()
#Work
sns.set(style="darkgrid")
sns.catplot(y="Work", x="Hours", data=df)
plt.show()
#Sagar
# Residency
Resident = df.groupby("Residence").count().reset_index()
print(Resident.columns)
plt.figure(figsize=(9,25))
ax = sns.barplot(x=Resident['S. No.'], y=Resident['Residence'],
                 data=Resident, palette="tab20c",
                 linewidth = 1)
for i,j in enumerate(Resident["S. No."]):
    ax.text(.5, i, j, weight="bold", color = 'black', fontsize =10)
plt.title("Population of each country in 2020")
ax.set_xlabel(xlabel = 'Population in Billion', fontsize = 10)
ax.set_ylabel(ylabel = 'Countries', fontsize = 10)
plt.show()
###Work

###Degree

#Sagar
df2=df.copy()
replace_map = {'Game':{'Counter Strike':1,'Destiny':2,'Diablo 3':3,'Guild Wars 2':4,'Hearthstone':5,'Heroes of the Storm':6,'League of Legends':7,'Other':8,'Skyrim':9,'Starcraft 2':10,'World of Warcraft':11},
               'GADE':{'Extremely difficult':3,'Very difficult':2,'Somewhat difficult':1,'Not difficult at all':0},
               'Platform':{'Console (PS, Xbox, ...)':0,'PC':1,'Smartphone / Tablet':2},
               'Gender':{'Male':0,'Female':1,'Other':2},
               'Work':{'Employed':0,'Unemployed / between jobs':1,'Student at college / university':2,'Student at school':3},
               'Degree':{'None':0,'High school diploma (or equivalent)':1,'Bachelor�(or equivalent)':2,'Master�(or equivalent)':3,'Ph.D., Psy. D., MD (or equivalent)':4},
               'Residence':{'Albania':0,'Algeria':1,'Argentina':2,'Australia':3,'Austria':4,'Bahrain':5,'Bangladesh':6,'Belarus':7,'Belgium':8,'Belize':9,'Bolivia':10,'Bosnia and Herzegovina':11,'Brazil':12,'Brunei':13,'Bulgaria':14,'Canada':15,'Chile':16,'China':17,'Colombia':18,'Costa Rica':19,'Croatia':20,'Cyprus':21,'Czech Republic':22,'Denmark':23,'Dominican Republic':24,'Ecuador':25,'Egypt':26,'El Salvador':27,'Estonia':28,'Faroe Islands':29,'Fiji':30,'Finland':31,'France':32,'Georgia':33,'Germany':34,'Gibraltar ':35,'Greece':36,'Grenada':37,'Guadeloupe':38,'Guatemala':39,'Honduras':40,'Hong Kong':41,'Hungary':42,'Iceland':43,'India':44,'India':45,'Indonesia':46,'Ireland':47,'Israel':48,'Italy':49,'Jamaica':50,'Japan':51,'Jordan':52,'Kazakhstan':53,'Kuwait':54,'Latvia':55,'Lebanon':56,'Liechtenstein':57,'Lithuania':58,'Luxembourg':59,'Macedonia':60,'Malaysia':61,'Malta':62,'Mexico':63,'Moldova':64,'Mongolia':65,'Montenegro':66,'Morocco':67,'Namibia':68,'Netherlands':69,'New Zealand ':70,'Nicaragua':71,'Norway':72,'Pakistan':73,'Palestine':74,'Panama':75,'Peru':76,'Philippines':77,'Poland':78,'Portugal':79,'Puerto Rico':80,'Qatar':81,'Republic of Kosovo':82,'Romania':83,'Russia':84,'Saudi Arabia':85,'Serbia':86,'Singapore':87,'Slovakia':88,'Slovenia':89,'South Africa':90,'South Korea':91,'Spain':92,'St Vincent':93,'Sweden':94,'Switzerland':95,'Syria':96,'Taiwan':97,'Thailand':98,'Trinidad & Tobago':99,'Tunisia':100,'Turkey':101,'UAE':102,'UK':103,'Ukraine':104,'Unknown':105,'Uruguay':106,'USA':107,'Venezuela':108,'Vietnam':109},
               'GAD_T':{'minimal anxiety':0, 'mild anxiety':1, 'moderate anxiety':2},
               'Playstyle':{'Singleplayer':0,'Multiplayer - online - with strangers':1,'Multiplayer - online - with online acquaintances or teammates':2,'Multiplayer - online - with real life friends':3,'all of the above':4,'Multiplayer - offline (people in the same room)':5}}

#,'severe anxiety':3
ax = sns.countplot(x="GAD_T", data=df2)
plt.show()

df2.replace(replace_map, inplace=True)

######### Modeling
# from sklearn.datasets import load_iris
# from sklearn import tree
# from sklearn.tree import export_graphviz
# import graphviz
#
# #countplot on Narcissism
#
# # decision Tree by MB
cols = df2[['GAD5','GAD6','GADE','SPIN_T','SWL_T','Game','Playstyle','Platform', 'Gender','Age','Hours','Work','Residence']]
x = cols.values
y = df2['GAD_T'].values
from sklearn.preprocessing import label_binarize
class_le = LabelEncoder()

y = class_le.fit_transform(y)

y1 = label_binarize(y, classes=[0,1,2])

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.3, random_state=1)
x_train1, x_test1, y_train1, y_test1 = train_test_split (x, y1, test_size=0.3, random_state=1)

# Fit dt to the training set
rf1 = DecisionTreeClassifier(max_depth=3,criterion='entropy',random_state=0)
# Fit dt to the training set
rf1.fit(x_train,y_train)
# y_train_pred = rf1.predict(x_train)
y_test_pred = rf1.predict(x_test)
y_pred_score = rf1.predict_proba(x_test)


rf2 = OneVsRestClassifier(DecisionTreeClassifier(max_depth=3,criterion='entropy'))
# Fit dt to the training set
rf2.fit(x_train1,y_train1)
# y_train_pred = rf1.predict(x_train)
y_test_pred1 = rf2.predict(x_test1)
y_pred_score1 = rf2.predict_proba(x_test1)

print('Decision Tree results')

# Evaluate test-set accuracy
print('test set evaluation: ')
print("Accuracy score: ",accuracy_score(y_test, y_test_pred)*100)
print("Confusion Matrix: \n",confusion_matrix(y_test, y_test_pred))
print("Classification report:\n",classification_report(y_test, y_test_pred))

from sklearn.metrics import roc_curve, auc

n_classes=3
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test1[:, i], y_pred_score1[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(f'AUC value of {i} class:{roc_auc[i]}')

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Decision Tree ROC')
    plt.legend(loc="lower right")
    plt.show()


# # Tree depth & leafs
# print ('Tree Depth:', dtc.get_depth())
# print ('Tree Leaves:', dtc.get_n_leaves())
#
#
#
# # # Graphing the tree
# from sklearn.tree import export_graphviz
#
# #export the decision tree to a tree.dot file
# # for visualizing the plot easily anywhere
#
# filename = 'tree1'
# import os
# # print(os.getcwd())
# export_graphviz(clf2, out_file = filename + '.dot' , feature_names =cols.columns[:])
#
# #%%
# import pydot
# (graph,) = pydot.graph_from_dot_file('tree1.dot')
# graph.write_png(filename+'.png')
#

#========================== random forest =========================
#%%
#Gresh

from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
# Instantiate dtree
rf1 = RandomForestClassifier(n_estimators=100)
# Fit dt to the training set
rf1.fit(x_train,y_train)
# y_train_pred = rf1.predict(x_train)
y_test_pred = rf1.predict(x_test)
y_pred_score = rf1.predict_proba(x_test)


rf2 = OneVsRestClassifier(RandomForestClassifier(n_estimators=100))
# Fit dt to the training set
rf2.fit(x_train1,y_train1)
# y_train_pred = rf1.predict(x_train)
y_test_pred1 = rf2.predict(x_test1)
y_pred_score1 = rf2.predict_proba(x_test1)

print('Random forest results')

# Evaluate test-set accuracy
print('test set evaluation: ')
print("Accuracy score: ",accuracy_score(y_test, y_test_pred)*100)
print("Confusion Matrix: \n",confusion_matrix(y_test, y_test_pred))
print("Classification report:\n",classification_report(y_test, y_test_pred))

from sklearn.metrics import roc_curve, auc

n_classes=3
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test1[:, i], y_pred_score1[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(f'AUC value of {i} class:{roc_auc[i]}')

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random Forest ROC')
    plt.legend(loc="lower right")
    plt.show()

#========================== SVC ============================
#%%
# #Asish
from sklearn.svm import SVC

rf1 = SVC(kernel='linear', C=1.0, random_state=0)
# Fit dt to the training set
rf1.fit(x_train,y_train)
# y_train_pred = rf1.predict(x_train)
y_test_pred = rf1.predict(x_test)
y_pred_score = rf1.decision_function(x_test)

rf2 = OneVsRestClassifier(SVC(kernel='linear', C=1.0, random_state=0))
# Fit dt to the training set
rf2.fit(x_train1,y_train1)
# y_train_pred = rf1.predict(x_train)
y_test_pred1 = rf2.predict(x_test1)
y_pred_score1 = rf2.decision_function(x_test1)


print('SVC results')

# Evaluate test-set accuracy
print('test set evaluation: ')
print("Accuracy score: ",accuracy_score(y_test, y_test_pred)*100)
print("Confusion Matrix: \n",confusion_matrix(y_test, y_test_pred))
print("Classification report:\n",classification_report(y_test, y_test_pred))

from sklearn.metrics import roc_curve, auc

n_classes=3
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test1[:, i], y_pred_score1[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(f'AUC value of {i} class:{roc_auc[i]}')

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVC ROC')
    plt.legend(loc="lower right")
    plt.show()