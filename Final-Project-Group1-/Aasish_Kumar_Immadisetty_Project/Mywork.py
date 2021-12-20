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

# Login to Kaggle and retrieve the data.
r = requests.post(login_url, data=kaggle_info, stream=True)
df = pd.read_csv('GamingStudy_data.csv', encoding='cp1252')
print(df)

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

#SVC
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

    #################PYQTGUI

class SupportVector(QMainWindow):
    #::----------------------
    # Implementation of Decision Tree Algorithm using the happiness dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parametes
    #               chosen by the user
    #       view_tree : shows the tree in a pdf form
    #::----------------------

    send_fig = pyqtSignal(str)

    def __init__(self):
        super(SupportVector, self).__init__()

        self.Title ="Support Vector Classifier"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid loyout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('ML Support Vector Features')
        self.groupBox1Layout= QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature0 = QCheckBox(features_list[0],self)
        self.feature1 = QCheckBox(features_list[1],self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4],self)
        self.feature5 = QCheckBox(features_list[5],self)
        self.feature6 = QCheckBox(features_list[6], self)
        self.feature7 = QCheckBox(features_list[7], self)
        self.feature8 = QCheckBox(features_list[8], self)
        self.feature9 = QCheckBox(features_list[9],self)
        self.feature10 = QCheckBox(features_list[10],self)
        self.feature11 = QCheckBox(features_list[11], self)
        self.feature12 = QCheckBox(features_list[12], self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)
        self.feature6.setChecked(True)
        self.feature7.setChecked(True)
        self.feature8.setChecked(True)
        self.feature9.setChecked(True)
        self.feature10.setChecked(True)
        self.feature11.setChecked(True)
        self.feature12.setChecked(True)

        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.lblMaxDepth = QLabel('Maximun Depth :')
        self.txtMaxDepth = QLineEdit(self)
        self.txtMaxDepth.setText("3")

        self.btnExecute = QPushButton("Execute SVM")
        self.btnExecute.clicked.connect(self.update)


        # We create a checkbox for each feature

        self.groupBox1Layout.addWidget(self.feature0,0,0)
        self.groupBox1Layout.addWidget(self.feature1,0,1)
        self.groupBox1Layout.addWidget(self.feature2,1,0)
        self.groupBox1Layout.addWidget(self.feature3,1,1)
        self.groupBox1Layout.addWidget(self.feature4,2,0)
        self.groupBox1Layout.addWidget(self.feature5,2,1)
        self.groupBox1Layout.addWidget(self.feature6,3,0)
        self.groupBox1Layout.addWidget(self.feature7,3,1)
        self.groupBox1Layout.addWidget(self.feature8,4,0)
        self.groupBox1Layout.addWidget(self.feature9,4,1)
        self.groupBox1Layout.addWidget(self.feature10,5,0)
        self.groupBox1Layout.addWidget(self.feature11,5,1)
        self.groupBox1Layout.addWidget(self.feature12,6,0)


        self.groupBox1Layout.addWidget(self.lblPercentTest,7,0)
        self.groupBox1Layout.addWidget(self.txtPercentTest,7,1)
        self.groupBox1Layout.addWidget(self.lblMaxDepth,8,0)
        self.groupBox1Layout.addWidget(self.txtMaxDepth,8,1)
        self.groupBox1Layout.addWidget(self.btnExecute,9,0)
        # self.groupBox1Layout.addWidget(self.btnDTFigure,9,1)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblAccuracy = QLabel('Accuracy:')
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)

        #::-------------------------------------
        # Graphic 1 : Confusion Matrix
        #::-------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        #::--------------------------------------------
        ## End Graph1
        #::--------------------------------------------

        #::---------------------------------------------
        # Graphic 2 : ROC Curve
        #::---------------------------------------------

        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas2.updateGeometry()

        self.groupBoxG2 = QGroupBox('ROC Curve')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)

        self.groupBoxG2Layout.addWidget(self.canvas2)

        #::---------------------------------------------------
        # Graphic 3 : ROC Curve by Class
        #::---------------------------------------------------

        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas3 = FigureCanvas(self.fig3)

        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas3.updateGeometry()

        self.groupBoxG3 = QGroupBox('ROC Curve by Class')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)

        self.groupBoxG3Layout.addWidget(self.canvas3)

        ## End of elements o the dashboard

        self.layout.addWidget(self.groupBox1,0,0)
        self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2,0,2)
        self.layout.addWidget(self.groupBoxG2,1,1)
        self.layout.addWidget(self.groupBoxG3,1,2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()


    def update(self):
        '''
        Decision Tree Algorithm
        We pupulate the dashboard using the parametres chosen by the user
        The parameters are processed to execute in the skit-learn Decision Tree algorithm
          then the results are presented in graphics and reports in the canvas
        :return: None
        '''

        # We process the parameters
        self.list_corr_features = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.list_corr_features)==0:
                self.list_corr_features = data[features_list[0]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, data[features_list[0]]],axis=1)

        if self.feature1.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = data[features_list[1]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, data[features_list[1]]],axis=1)

        if self.feature2.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = data[features_list[2]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, data[features_list[2]]],axis=1)

        if self.feature3.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = data[features_list[3]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, data[features_list[3]]],axis=1)

        if self.feature4.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = data[features_list[4]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, data[features_list[4]]],axis=1)

        if self.feature5.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = data[features_list[5]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, data[features_list[5]]],axis=1)

        if self.feature6.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = data[features_list[6]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, data[features_list[6]]],axis=1)

        if self.feature7.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = data[features_list[7]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, data[features_list[7]]],axis=1)

        if self.feature8.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = data[features_list[8]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, data[features_list[8]]],axis=1)

        if self.feature9.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = data[features_list[9]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, data[features_list[9]]],axis=1)

        if self.feature10.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = data[features_list[10]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, data[features_list[10]]],axis=1)

        if self.feature11.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = data[features_list[11]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, data[features_list[11]]],axis=1)

        if self.feature12.isChecked():
            if len(self.list_corr_features) == 0:
                self.list_corr_features = data[features_list[12]]
            else:
                self.list_corr_features = pd.concat([self.list_corr_features, data[features_list[12]]],axis=1)


        vtest_per = float(self.txtPercentTest.text())
        vmax_depth = float(self.txtMaxDepth.text())

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100


        # We assign the values to X and y to run the algorithm

        X_dt =  self.list_corr_features
        y_dt = data["GAD_T"]

        class_le = LabelEncoder()

        # fit and transform the class

        y_dt = class_le.fit_transform(y_dt)

        # split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=100)
        # perform training with entropy.
        # Decision tree with entropy
        self.clf_entropy = SVC(kernel='linear', C=1.0, random_state=0)

        # Performing training
        self.clf_entropy.fit(X_train, y_train)

        # predicton on test using entropy
        y_pred_entropy = self.clf_entropy.predict(X_test)

        # confusion matrix for entropy model

        conf_matrix = confusion_matrix(y_test, y_pred_entropy)

        # clasification report

        self.ff_class_rep = classification_report(y_test, y_pred_entropy)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # accuracy score

        self.ff_accuracy_score = accuracy_score(y_test, y_pred_entropy) * 100
        self.txtAccuracy.setText(str(self.ff_accuracy_score))


        #::----------------------------------------------------------------
        # Graph1 -- Confusion Matrix
        #::-----------------------------------------------------------------
        class_names1 = [0, 1, 2]

        self.ax1.matshow(conf_matrix, cmap=plt.cm.get_cmap('Blues', 14))
        self.ax1.set_ylabel(class_names1)
        self.ax1.set_xlabel(class_names1)

        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')


        for i in range(len(class_names1)):
            for j in range(len(class_names1)):
                y_pred_score = self.clf_entropy.decision_function(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        #::-----------------------------------------------------
        # End Graph 1 -- Confusion Matrix
        #::-----------------------------------------------------

        #::-----------------------------------------------------
        # Graph 2 -- ROC Cure
        #::-----------------------------------------------------

        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        n_classes = y_test_bin.shape[1]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_score.ravel())

        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        lw = 2
        self.ax2.plot(fpr[2], tpr[2], color='darkorange',
                      lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        self.ax2.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        self.ax2.set_xlim([0.0, 1.0])
        self.ax2.set_ylim([0.0, 1.05])
        self.ax2.set_xlabel('False Positive Rate')
        self.ax2.set_ylabel('True Positive Rate')
        self.ax2.set_title('ROC Curve by class')
        self.ax2.legend(loc="lower right")

        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

        #::--------------------------------
        ### Graph 3 Roc Curve by class
        #::--------------------------------



        str_classes= ['minimal','mild','moderate']
        colors = cycle(['magenta', 'darkorange', 'green', 'blue'])
        for i, color in zip(range(n_classes), colors):
            self.ax3.plot(fpr[i], tpr[i], color=color, lw=lw,
                          label='{0} (area = {1:0.2f})'
                                ''.format(str_classes[i], roc_auc[i]))

        self.ax3.plot([0, 1], [0, 1], 'k--', lw=lw)
        self.ax3.set_xlim([0.0, 1.0])
        self.ax3.set_ylim([0.0, 1.05])
        self.ax3.set_xlabel('False Positive Rate')
        self.ax3.set_ylabel('True Positive Rate')
        self.ax3.set_title('ROC Curve by Class')
        self.ax3.legend(loc="lower right")

        # show the plot
        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()

