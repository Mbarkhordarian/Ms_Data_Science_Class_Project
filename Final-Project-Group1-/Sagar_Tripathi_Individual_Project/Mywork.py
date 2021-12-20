from sklearn.svm import SVC
import sys
import requests
#from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel, QGridLayout, QCheckBox, QGroupBox
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel,
                             QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit)
from PyQt5.QtWidgets import QTableWidget,QTableWidgetItem, QHeaderView
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt
import warnings
from scipy import interpolate
from itertools import cycle
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
warnings.filterwarnings("ignore")
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon, QImage , QPalette , QBrush,QPixmap
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QSizePolicy, QMessageBox
from sklearn.preprocessing import label_binarize,LabelEncoder
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Libraries to display decision tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

import random
import seaborn as sns

#%%-----------------------------------------------------------------------
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\graphviz-2.38\\release\\bin'
#%%-----------------------------------------------------------------------


#::--------------------------------
# Deafault font size for all the windows
#::--------------------------------
font_size_window = 'font-size:15px'



# Link to the Kaggle data set & name of zip file
login_url = 'https://www.kaggle.com/divyansh22/online-gaming-anxiety-data?select=GamingStudy_data.csv'

# Kaggle Username and Password
kaggle_info = {'UserName': "iaasish123@gwmail.gwu.edu", 'Password': "Password@123"}

# Login to Kaggle and retrieve the data.
r = requests.post(login_url, data=kaggle_info, stream=True)
df = pd.read_csv('GamingStudy_data.csv')
print(df)

df["Age"] = pd.to_numeric(df["Age"])
df["Hours"] = pd.to_numeric(df["Hours"])
df["streams"] = pd.to_numeric(df["streams"])
df["Hours"]=pd.to_numeric(df['Hours'])
df["GAD_T"]=pd.to_numeric(df['GAD_T'])

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
df2=df.copy()
replace_map = {'Game':{'Counter Strike':1,'Destiny':2,'Diablo 3':3,'Guild Wars 2':4,'Hearthstone':5,'Heroes of the Storm':6,'League of Legends':7,'Other':8,'Skyrim':9,'Starcraft 2':10,'World of Warcraft':11},
               'GADE':{'Extremely difficult':3,'Very difficult':2,'Somewhat difficult':1,'Not difficult at all':0},
               'Platform':{'Console (PS, Xbox, ...)':0,'PC':1,'Smartphone / Tablet':2},
               'Gender':{'Male':0,'Female':1,'Other':2},
               'Work':{'Employed':0,'Unemployed / between jobs':1,'Student at college / university':2,'Student at school':3},
               'Degree':{'None':0,'High school diploma (or equivalent)':1,'Bachelor�(or equivalent)':2,'Master�(or equivalent)':3,'Ph.D., Psy. D., MD (or equivalent)':4},
               'Residence':{'Albania':0,'Algeria':1,'Argentina':2,'Australia':3,'Austria':4,'Bahrain':5,'Bangladesh':6,'Belarus':7,'Belgium':8,'Belize':9,'Bolivia':10,'Bosnia and Herzegovina':11,'Brazil':12,'Brunei':13,'Bulgaria':14,'Canada':15,'Chile':16,'China':17,'Colombia':18,'Costa Rica':19,'Croatia':20,'Cyprus':21,'Czech Republic':22,'Denmark':23,'Dominican Republic':24,'Ecuador':25,'Egypt':26,'El Salvador':27,'Estonia':28,'Faroe Islands':29,'Fiji':30,'Finland':31,'France':32,'Georgia':33,'Germany':34,'Gibraltar ':35,'Greece':36,'Grenada':37,'Guadeloupe':38,'Guatemala':39,'Honduras':40,'Hong Kong':41,'Hungary':42,'Iceland':43,'India':44,'India':45,'Indonesia':46,'Ireland':47,'Israel':48,'Italy':49,'Jamaica':50,'Japan':51,'Jordan':52,'Kazakhstan':53,'Kuwait':54,'Latvia':55,'Lebanon':56,'Liechtenstein':57,'Lithuania':58,'Luxembourg':59,'Macedonia':60,'Malaysia':61,'Malta':62,'Mexico':63,'Moldova':64,'Mongolia':65,'Montenegro':66,'Morocco':67,'Namibia':68,'Netherlands':69,'New Zealand ':70,'Nicaragua':71,'Norway':72,'Pakistan':73,'Palestine':74,'Panama':75,'Peru':76,'Philippines':77,'Poland':78,'Portugal':79,'Puerto Rico':80,'Qatar':81,'Republic of Kosovo':82,'Romania':83,'Russia':84,'Saudi Arabia':85,'Serbia':86,'Singapore':87,'Slovakia':88,'Slovenia':89,'South Africa':90,'South Korea':91,'Spain':92,'St Vincent':93,'Sweden':94,'Switzerland':95,'Syria':96,'Taiwan':97,'Thailand':98,'Trinidad & Tobago':99,'Tunisia':100,'Turkey':101,'UAE':102,'UK':103,'Ukraine':104,'Unknown':105,'Uruguay':106,'USA':107,'Venezuela':108,'Vietnam':109},
               'GAD_T':{'minimal anxiety':0, 'mild anxiety':1, 'moderate anxiety':2,'severe':3},
               'Playstyle':{'Singleplayer':0,'Multiplayer - online - with strangers':1,'Multiplayer - online - with online acquaintances or teammates':2,'Multiplayer - online - with real life friends':3,'all of the above':4,'Multiplayer - offline (people in the same room)':5}}

#MtPYQT



class PlotCanvas(FigureCanvas):
    #::----------------------------------------------------------
    # creates a figure on the canvas
    # later on this element will be used to draw a histogram graph
    #::----------------------------------------------------------
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self):
        self.ax = self.figure.add_subplot(111)

class CanvasWindow(QMainWindow):
    #::----------------------------------
    # Creates a canvaas containing the plot for the initial analysis
    #;;----------------------------------
    def __init__(self, parent=None):
        super(CanvasWindow, self).__init__(parent)

        self.left = 200
        self.top = 200
        self.Title = 'Distribution'
        self.width = 500
        self.height = 500
        self.initUI()

    def initUI(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.setGeometry(self.left, self.top, self.width, self.height)

        self.m = PlotCanvas(self, width=5, height=4)
        self.m.move(0, 30)

class Histogram_plots(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(Histogram_plots, self).__init__()
        self.Title = "Histograms"
        self.initUi()

    def initUi(self):

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Select One of the Features')
        self.groupBox1Layout = QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature = []

        for i in range(13):
            self.feature.append(QCheckBox(features_list_hist[i], self))

        for i in self.feature:
            i.setChecked(False)

        self.btnExecute = QPushButton("Plot")

        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.feature[0], 0, 0)
        self.groupBox1Layout.addWidget(self.feature[1], 0, 1)
        self.groupBox1Layout.addWidget(self.feature[2], 1, 0)
        self.groupBox1Layout.addWidget(self.feature[3], 1, 1)
        self.groupBox1Layout.addWidget(self.feature[4], 2, 0)
        self.groupBox1Layout.addWidget(self.feature[5], 2, 1)
        self.groupBox1Layout.addWidget(self.feature[6], 3, 0)
        self.groupBox1Layout.addWidget(self.feature[7], 3, 1)
        self.groupBox1Layout.addWidget(self.feature[8], 4, 0)
        self.groupBox1Layout.addWidget(self.feature[9], 4, 1)
        self.groupBox1Layout.addWidget(self.feature[10], 5, 0)
        self.groupBox1Layout.addWidget(self.feature[11], 5, 1)
        self.groupBox1Layout.addWidget(self.feature[12], 6, 0)
        self.groupBox1Layout.addWidget(self.btnExecute, 7, 1)

        self.fig1, self.ax1 = plt.subplots()
        self.axes = [self.ax1]
        self.canvas1 = FigureCanvas(self.fig1)

        self.canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas1.updateGeometry()

        self.groupBoxG1 = QGroupBox('Histogram Plot :')
        self.groupBoxG1Layout = QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas1)

        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBoxG1, 0, 1)

        self.setCentralWidget(self.main_widget)
        self.resize(1200, 900)
        self.show()

    def Message(self):
        QMessageBox.about(self, "Warning", " You can't exceed more than 1 feature")

    def update(self):
        self.current_features = pd.DataFrame([])
        x_a = ''
        work = 0
        for i in range(13):
            if self.feature[i].isChecked():
                if len(self.current_features) > 1:
                    self.Message()
                    work = 1
                    break

                elif len(self.current_features) == 0:
                    self.current_features = data[features_list_hist[i]]
                    x_a = features_list_hist[i]
                    work=0

        if work == 0:
            self.ax1.clear()
            self.current_features.value_counts().plot(kind='bar', ax=self.ax1)
            self.ax1.set_title('Histogram of : ' + x_a)
            self.ax1.set_xlabel(x_a)
            self.ax1.set_ylabel('frequency')
            self.fig1.tight_layout()
            self.fig1.canvas.draw_idle()

class App(QMainWindow):
    #::-------------------------------------------------------
    # This class creates all the elements of the application
    #::-------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.left = 100
        self.top = 100
        self.Title = 'Generalized Anxiety Disorder '
        self.width = 800
        self.height = 300
        self.initUI()

    def initUI(self):
        #::-------------------------------------------------
        # Creates the manu and the items
        #::-------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #::-----------------------------
        # Create the menu bar
        # and three items for the menu, File, EDA Analysis and ML Models
        #::-----------------------------
        mainMenu = self.menuBar()
        mainMenu.setStyleSheet("color: white;"
                               "background-color: black;"
                               "selection-color: black;"
                               "selection-background-color: white;")


        label1 = QLabel(self)
        label1.setText("<font color = black>Generalized Anxiety Disorder Application</font>")
        label1.setFont(QtGui.QFont("Times", 16, QtGui.QFont.Bold))
        label1.move(200, 5)
        label1.resize(400, 350)

        fileMenu = mainMenu.addMenu('File')
        EDAMenu = mainMenu.addMenu('EDA Analysis')
        MLModelMenu = mainMenu.addMenu('ML Models')

        #::--------------------------------------
        # Exit application
        # Creates the actions for the fileMenu item
        #::--------------------------------------

        exitButton = QAction(QIcon('enter.png'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)

        #::----------------------------------------
        # EDA analysis
        # Creates the actions for the EDA Analysis item
        # Initial Assesment : Histogram about the level of happiness in 2017
        # Happiness Final : Presents the correlation between the index of happiness and a feature from the datasets.
        # Correlation Plot : Correlation plot using all the dims in the datasets
        #::----------------------------------------

        EDA1Button = QAction(QIcon('analysis.png'),'Initial Assesment', self)
        EDA1Button.setStatusTip('Presents the initial datasets')
        EDA1Button.triggered.connect(self.EDA1)
        EDAMenu.addAction(EDA1Button)

        EDA2Button = QAction(QIcon('analysis.png'), 'Scatter plot', self)
        EDA2Button.setStatusTip('Final Happiness Graph')
        EDA2Button.triggered.connect(self.EDA2)
        EDAMenu.addAction(EDA2Button)

        EDA4Button = QAction(QIcon('analysis.png'), 'Correlation Plot', self)
        EDA4Button.setStatusTip('Features Correlation Plot')
        EDA4Button.triggered.connect(self.EDA4)
        EDAMenu.addAction(EDA4Button)

        #::--------------------------------------------------
        # ML Models for prediction
        # There are two models
        #       Decision Tree
        #       Random Forest
        #      Support Vector Machine
        #::--------------------------------------------------
        # Decision Tree Model
        #::--------------------------------------------------
        MLModel1Button =  QAction(QIcon(), 'Decision Tree Entropy', self)
        MLModel1Button.setStatusTip('ML algorithm with Entropy ')
        MLModel1Button.triggered.connect(self.MLDT)

        #::------------------------------------------------------
        # Random Forest Classifier
        #::------------------------------------------------------
        MLModel2Button = QAction(QIcon(), 'Random Forest Classifier', self)
        MLModel2Button.setStatusTip('Random Forest Classifier ')
        MLModel2Button.triggered.connect(self.MLRF)

        #::------------------------------------------------------
        # Support Vector Machine
        #::------------------------------------------------------
        MLModel3Button = QAction(QIcon(), 'Support Vector Machine', self)
        MLModel3Button.setStatusTip('Support Vector Machine ')
        MLModel3Button.triggered.connect(self.MLSVM)

        MLModelMenu.addAction(MLModel1Button)
        MLModelMenu.addAction(MLModel2Button)
        MLModelMenu.addAction(MLModel3Button)

        self.dialogs = list()




    def EDA1(self):
        #::------------------------------------------------------
        # Creates the Histogram plot
        #::------------------------------------------------------

        dialog = Histogram_plots()
        self.dialogs.append(dialog)
        dialog.show()


    def EDA2(self):
        #::------------------------------------------------------
        # Creates the scatter plot
        #::------------------------------------------------------
        dialog = HappinessGraphs()
        self.dialogs.append(dialog)
        dialog.show()

    def EDA4(self):
        #::----------------------------------------------------------
        # This function creates an instance of the CorrelationPlot class
        #::----------------------------------------------------------
        dialog = CorrelationPlot()
        self.dialogs.append(dialog)
        dialog.show()

    def MLDT(self):
        #::-----------------------------------------------------------
        # This function creates an instance of the DecisionTree class
        # using the Generalized Anxiety Disorder dataset
        #::-----------------------------------------------------------
        dialog = DecisionTree()
        self.dialogs.append(dialog)
        dialog.show()

    def MLRF(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Random Forest Classifier Algorithm
        # using the happiness dataset
        #::-------------------------------------------------------------
        dialog = RandomForest()
        self.dialogs.append(dialog)
        dialog.show()


    def MLSVM(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Support Vector Classifier Algorithm
        # using the Generalized Anxiety Disorder dataset
        #::-------------------------------------------------------------
        dialog = SupportVector()
        self.dialogs.append(dialog)
        dialog.show()
    def Message_up(self):
        QMessageBox.about(self, "Warning", " You have not Uploaded the data")


def main():
    #::-------------------------------------------------
    # Initiates the application
    #::-------------------------------------------------
    app = QApplication(sys.argv)
    app.setStyle('Windows')
    ex = App()
    ex.show()
    sys.exit(app.exec_())


def data_happiness():
    #::--------------------------------------------------
    # Loads the dataset 2017.csv ( Index of happiness and esplanatory variables original dataset)
    # Loads the dataset final_happiness_dataset (index of happiness
    # and explanatory variables which are already preprocessed)
    # Populates X,y that are used in the classes above
    #::--------------------------------------------------

    global data
    global features_list
    global class_names
    global features_list_hist
    global r
    global replace_map
    global X
    global Y
    global login_url
    global kaggle_info
    global data
    global features_list_hist
    global class_le
    # Link to the Kaggle data set & name of zip file
    login_url = 'https://www.kaggle.com/divyansh22/online-gaming-anxiety-data?select=GamingStudy_data.csv'

    # Kaggle Username and Password
    kaggle_info = {'UserName': "iaasish123@gwmail.gwu.edu", 'Password': "Password@123"}
    r = requests.post(login_url, data=kaggle_info, stream=True)
    data = pd.read_csv('GamingStudy_data.csv',encoding='cp1252')

    data.drop(['Narcissism','streams','SPIN1','SPIN2','SPIN3','SPIN4','SPIN5','SPIN6','SPIN7','SPIN8','SPIN9','SPIN10','SPIN11','SPIN12','SPIN13','SPIN14','SPIN15','SPIN16','SPIN17','Timestamp','accept','League','Birthplace','Reference','Birthplace_ISO3','highestleague','SWL1','SWL2','SWL3','SWL4','SWL5','earnings','whyplay','Birthplace_ISO3','Residence_ISO3'], axis=1, inplace=True)
    data.drop(data[(data['Playstyle']!='Singleplayer') & (data['Playstyle']!='Multiplayer - online - with strangers') & (data['Playstyle']!='Multiplayer - online - with online acquaintances or teammates') & (data['Playstyle']!='Multiplayer - online - with real life friends') & (data['Playstyle']!='Multiplayer - offline (people in the same room)') & (data['Playstyle']!='all of the above')].index,axis=0,inplace=True)
    data = data.apply(lambda x: x.fillna(x.value_counts().index[0]))

    #convert the variables to numeric
    data["Age"] = pd.to_numeric(data["Age"])
    data["Hours"] = pd.to_numeric(data["Hours"])
    data["GAD_T"]=pd.to_numeric(data['GAD_T'])
    # droping the outliers
    data = data[(-3 < zscore(data['Hours'])) & (zscore(data['Hours']) < 3)]
    data = data[(-3 < zscore(data['Age'])) & (zscore(data['Age']) < 3)]
    data = data[(-3 < zscore(data['GAD_T'])) & (zscore(data['GAD_T']) < 3)]
    data = data[(-3 < zscore(data['SWL_T'])) & (zscore(data['SWL_T']) < 3)]

    gad_new = []
    for i in data['GAD_T']:
        if i <= 4:
            gad_new.append('mild')
        elif ((i >= 5) & (i <= 9)):
            gad_new.append('moderate')
        elif (i >= 10):
            # &(i<=14)):
            gad_new.append('moderately severe')
        # elif i>=15:
        # gad_new.append('severe')
    data['GAD_T'] = gad_new



    data=data.copy()
    replace_map = {'Game':{'Counter Strike':1,'Destiny':2,'Diablo 3':3,'Guild Wars 2':4,'Hearthstone':5,'Heroes of the Storm':6,'League of Legends':7,'Other':8,'Skyrim':9,'Starcraft 2':10,'World of Warcraft':11},
                   'GADE':{'Extremely difficult':3,'Very difficult':2,'Somewhat difficult':1,'Not difficult at all':0},
                   'Platform':{'Console (PS, Xbox, ...)':0,'PC':1,'Smartphone / Tablet':2},
                   'Gender':{'Male':0,'Female':1,'Other':2},
                   'Work':{'Employed':0,'Unemployed / between jobs':1,'Student at college / university':2,'Student at school':3},
                   'Degree':{'None':0,'High school diploma (or equivalent)':1,'Bachelor�(or equivalent)':2,'Master�(or equivalent)':3,'Ph.D., Psy. D., MD (or equivalent)':4},
                   'Residence':{'Albania':0,'Algeria':1,'Argentina':2,'Australia':3,'Austria':4,'Bahrain':5,'Bangladesh':6,'Belarus':7,'Belgium':8,'Belize':9,'Bolivia':10,'Bosnia and Herzegovina':11,'Brazil':12,'Brunei':13,'Bulgaria':14,'Canada':15,'Chile':16,'China':17,'Colombia':18,'Costa Rica':19,'Croatia':20,'Cyprus':21,'Czech Republic':22,'Denmark':23,'Dominican Republic':24,'Ecuador':25,'Egypt':26,'El Salvador':27,'Estonia':28,'Faroe Islands':29,'Fiji':30,'Finland':31,'France':32,'Georgia':33,'Germany':34,'Gibraltar ':35,'Greece':36,'Grenada':37,'Guadeloupe':38,'Guatemala':39,'Honduras':40,'Hong Kong':41,'Hungary':42,'Iceland':43,'India':44,'India':45,'Indonesia':46,'Ireland':47,'Israel':48,'Italy':49,'Jamaica':50,'Japan':51,'Jordan':52,'Kazakhstan':53,'Kuwait':54,'Latvia':55,'Lebanon':56,'Liechtenstein':57,'Lithuania':58,'Luxembourg':59,'Macedonia':60,'Malaysia':61,'Malta':62,'Mexico':63,'Moldova':64,'Mongolia':65,'Montenegro':66,'Morocco':67,'Namibia':68,'Netherlands':69,'New Zealand ':70,'Nicaragua':71,'Norway':72,'Pakistan':73,'Palestine':74,'Panama':75,'Peru':76,'Philippines':77,'Poland':78,'Portugal':79,'Puerto Rico':80,'Qatar':81,'Republic of Kosovo':82,'Romania':83,'Russia':84,'Saudi Arabia':85,'Serbia':86,'Singapore':87,'Slovakia':88,'Slovenia':89,'South Africa':90,'South Korea':91,'Spain':92,'St Vincent':93,'Sweden':94,'Switzerland':95,'Syria':96,'Taiwan':97,'Thailand':98,'Trinidad & Tobago':99,'Tunisia':100,'Turkey':101,'UAE':102,'UK':103,'Ukraine':104,'Unknown':105,'Uruguay':106,'USA':107,'Venezuela':108,'Vietnam':109},
                   'GAD_T':{'minimal anxiety':0, 'mild anxiety':1, 'moderate anxiety':2},
                   'Playstyle':{'Singleplayer':0,'Multiplayer - online - with strangers':1,'Multiplayer - online - with online acquaintances or teammates':2,'Multiplayer - online - with real life friends':3,'all of the above':4,'Multiplayer - offline (people in the same room)':5}}

    data.replace(replace_map, inplace=True)

    features_list = ['GAD5','GAD6','GADE','SPIN_T','SWL_T','Game','Playstyle','Platform', 'Gender','Age','Hours','Work','Residence']
    features_list1 = data[['GAD5','GAD6','GADE','SPIN_T','SWL_T','Game','Playstyle','Platform', 'Gender','Age','Hours','Work','Residence']]
    X = features_list1.values
    y = data['GAD_T'].values
    features_list_hist = features_list
    class_le = LabelEncoder()
    class_names = class_le.fit_transform(y)

if __name__ == '__main__':
    #::------------------------------------
    # First reads the data then calls for the application
    #::------------------------------------
    data_happiness()
    main()