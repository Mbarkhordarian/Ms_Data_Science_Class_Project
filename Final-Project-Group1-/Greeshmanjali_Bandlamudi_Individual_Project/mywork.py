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
df = pd.read_csv('GamingStudy_data.csv')
print(df)
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

    ########MYPIQT
    class RandomForest(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Implementation of Random Forest Classifier using the happiness dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parametes
    #               chosen by the user
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(RandomForest, self).__init__()
        self.Title = "Radom Forest Classifier"
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

        self.groupBox1 = QGroupBox('ML Random Forest Features')
        self.groupBox1Layout= QGridLayout()   # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)

        # We create a checkbox of each Features
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



        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.btnExecute = QPushButton("Execute RF")
        self.btnExecute.clicked.connect(self.update)
        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.groupBox1Layout.addWidget(self.feature0,0,0)
        self.groupBox1Layout.addWidget(self.feature1,0,1)
        self.groupBox1Layout.addWidget(self.feature2,1,0)
        self.groupBox1Layout.addWidget(self.feature3,1,1)
        self.groupBox1Layout.addWidget(self.feature4,2,0)
        self.groupBox1Layout.addWidget(self.feature5,2,1)
        self.groupBox1Layout.addWidget(self.feature6,3,0)
        self.groupBox1Layout.addWidget(self.feature7,3,1)
        self.groupBox1Layout.addWidget(self.feature8,4,1)
        self.groupBox1Layout.addWidget(self.feature9,4,0)
        self.groupBox1Layout.addWidget(self.feature10,5,1)
        self.groupBox1Layout.addWidget(self.feature11,5,0)
        self.groupBox1Layout.addWidget(self.feature12,6,1)
        self.groupBox1Layout.addWidget(self.lblPercentTest,7,0)
        self.groupBox1Layout.addWidget(self.txtPercentTest,7,1)
        self.groupBox1Layout.addWidget(self.btnExecute,8,0)


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

        #::--------------------------------------
        # Graphic 1 : Confusion Matrix
        #::--------------------------------------

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

        #::---------------------------------------
        # Graphic 2 : ROC Curve
        #::---------------------------------------

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

        #::-------------------------------------------
        # Graphic 3 : Importance of Features
        #::-------------------------------------------

        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas3 = FigureCanvas(self.fig3)

        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas3.updateGeometry()

        self.groupBoxG3 = QGroupBox('Importance of Features')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)
        self.groupBoxG3Layout.addWidget(self.canvas3)

        #::--------------------------------------------
        # Graphic 4 : ROC Curve by class
        #::--------------------------------------------

        self.fig4 = Figure()
        self.ax4 = self.fig4.add_subplot(111)
        self.axes4 = [self.ax4]
        self.canvas4 = FigureCanvas(self.fig4)

        self.canvas4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas4.updateGeometry()

        self.groupBoxG4 = QGroupBox('ROC Curve by Class')
        self.groupBoxG4Layout = QVBoxLayout()
        self.groupBoxG4.setLayout(self.groupBoxG4Layout)
        self.groupBoxG4Layout.addWidget(self.canvas4)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1,0,0)
        self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2,1,0)
        self.layout.addWidget(self.groupBoxG2,1,1)
        self.layout.addWidget(self.groupBoxG3,0,2)
        self.layout.addWidget(self.groupBoxG4,1,2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        '''
        Random Forest Classifier
        We pupulate the dashboard using the parametres chosen by the user
        The parameters are processed to execute in the skit-learn Random Forest algorithm
          then the results are presented in graphics and reports in the canvas
        :return:None
        '''

        # processing the parameters

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

        # Clear the graphs to populate them with the new information

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100

        # Assign the X and y to run the Random Forest Classifier

        X_dt =  self.list_corr_features
        y_dt = data["GAD_T"]

        class_le = LabelEncoder()

        # fit and transform the class

        y_dt = class_le.fit_transform(y_dt)

        # split the dataset into train and test

        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=100)

        # perform training with entropy.
        # Decision tree with entropy

        #specify random forest classifier
        self.clf_rf = RandomForestClassifier(n_estimators=100, random_state=100)

        # perform training
        self.clf_rf.fit(X_train, y_train)

        #-----------------------------------------------------------------------

        # predicton on test using all features
        y_pred = self.clf_rf.predict(X_test)
        y_pred_score = self.clf_rf.predict_proba(X_test)


        # confusion matrix for RandomForest
        conf_matrix = confusion_matrix(y_test, y_pred)

        # clasification report

        self.ff_class_rep = classification_report(y_test, y_pred)
        self.txtResults.appendPlainText(self.ff_class_rep)

        # accuracy score

        self.ff_accuracy_score = accuracy_score(y_test, y_pred) * 100
        self.txtAccuracy.setText(str(self.ff_accuracy_score))

        #::------------------------------------
        ##  Ghaph1 :
        ##  Confusion Matrix
        #::------------------------------------
        # class_names1 = label_binarize(class_names, classes=[0,1,2])
        class_names1 = [0,1,2]

        self.ax1.matshow(conf_matrix, cmap= plt.cm.get_cmap('Blues', 14))
        self.ax1.set_ylabel(class_names1)
        self.ax1.set_xlabel(class_names1)
        # self.ax1.set_yticklabels(class_names1)
        # self.ax1.set_xticklabels(class_names1,rotation = 90)
        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')

        for i in range(len(class_names1)):
            for j in range(len(class_names1)):
                y_pred_score = self.clf_rf.predict_proba(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        ## End Graph1 -- Confusion Matrix

        #::----------------------------------------
        ## Graph 2 - ROC Curve
        #::----------------------------------------
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        n_classes = y_test_bin.shape[1]

        #From the sckict learn site
        #https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
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
        self.ax2.set_title('ROC Curve Random Forest')
        self.ax2.legend(loc="lower right")

        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()
        # ######################################
        # # Graph - 3 Feature Importances
        # #####################################
        # get feature importances
        importances = self.clf_rf.feature_importances_

        # convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
        f_importances = pd.Series(importances, self.list_corr_features.columns)

        # sort the array in descending order of the importances
        f_importances.sort_values(ascending=False, inplace=True)

        X_Features = f_importances.index
        y_Importance = list(f_importances)

        self.ax3.barh(X_Features, y_Importance )
        self.ax3.set_aspect('auto')

        # show the plot
        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()

        #::-----------------------------------------------------
        # Graph 4 - ROC Curve by Class
        #::-----------------------------------------------------
        str_classes= ['minimal','mild','moderate']
        colors = cycle(['magenta', 'darkorange', 'green', 'blue'])
        for i, color in zip(range(n_classes), colors):
            self.ax4.plot(fpr[i], tpr[i], color=color, lw=lw,
                          label='{0} (area = {1:0.2f})'
                                ''.format(str_classes[i], roc_auc[i]))

        self.ax4.plot([0, 1], [0, 1], 'k--', lw=lw)
        self.ax4.set_xlim([0.0, 1.0])
        self.ax4.set_ylim([0.0, 1.05])
        self.ax4.set_xlabel('False Positive Rate')
        self.ax4.set_ylabel('True Positive Rate')
        self.ax4.set_title('ROC Curve by Class')
        self.ax4.legend(loc="lower right")

        # show the plot
        self.fig4.tight_layout()
        self.fig4.canvas.draw_idle()

        #::-----------------------------
        # End of graph 4  - ROC curve by class
        #::-----------------------------
