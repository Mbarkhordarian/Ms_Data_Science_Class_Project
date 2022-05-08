import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import os
import numpy as np
import seaborn as sns
import statsmodels
import matplotlib
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.arima_model
from sklearn.model_selection import train_test_split
from pandas import Series
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn import metrics
from numpy import linalg as LA
import math
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('underwater_temperature.csv',index_col='ID', parse_dates=True,encoding= 'unicode_escape')
z=data.copy(deep=True)
df=z.iloc[::3,:]

print(df.isnull().sum().sum())
print(df.columns)
#Preprocessing
print(df['Site'].isnull().sum().sum())
print(df['Latitude'].isnull().sum().sum())
print(df['Longitude'].isnull().sum().sum())
print(df['Date'].isnull().sum().sum())
print(df['Time'].isnull().sum().sum())
print(df['Temp (°C)'].isnull().sum().sum()) #2 missing values

#preproccesing the Data›
df.rename(columns={'Temp (°C)':'Temp'},inplace=True)
mean_value=df['Temp'].mean()
df['Temp'].fillna(value=mean_value, inplace=True)
print(df['Temp'].isnull().sum().sum()) #2 missing values

#now we don't have any missing values
#plot the data vs time
sns.lineplot(data=df, x="Time",y="Temp")
plt.title('the dependency of temp and time')
plt.show()
#ACF/PACF of the dependent variable
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
def ACF_PACF_Plot(y,lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()
ACF_PACF_Plot(df['Temp'],20)
print(pd.unique(df['Site']))
#because of PACF the Ar is going to be 1

corr=df.corr()
ax =sns.heatmap(corr)
plt.show()
df['Time']=df["Time"].str[:-3]
y=df['Temp']
x=df.drop(['Temp','Site','Date','Time'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#rolling mean
def rolling_mean_var(x):
    df_rolling = pd.DataFrame()
    mean=[]
    var=[]
    for i in range(len(x)):
        mean.append(x[0:i+1].mean())
        var.append(x[0:i+1].var())
    df_rolling['mean'] = mean
    df_rolling['var'] = var
    df_rolling.fillna(0, inplace=True)
    #====
    fig = plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot('mean', data=df_rolling, label="Mean", color='red')
    plt.title("Rolling Mean & Variance")
    plt.xticks([])
    plt.grid()
    plt.ylabel('mean')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot('var', data=df_rolling, label="variance", color='blue')
    plt.grid()
    plt.ylabel('variance')
    plt.legend()
    plt.show()
rolling_mean_var(df['Temp'])

#Stationary
#ADF test for temp

x=df['Temp'].values
resault=adfuller(x)
print('ADF Statistic :%f'% resault[0])
print('p_value: %f'% resault[1])
print('Critical Values:')
for key, value in resault[4].items():
    print('\t%s:%.3f'%(key,value))
if resault[0] < resault[4]['5%']:
    print('Temp is Stationary')
else:
    print('Temp is not stationory')
#KPS Test
from statsmodels.tsa.stattools import kpss
x=df['Temp'].values
def kpss_test(x):
    print ('Results of KPSS Test:')
kpsstest = kpss(x, regression='c', nlags="auto")
kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','LagsUsed'])
for key,value in kpsstest[3].items():
    kpss_output['Critical Value (%s)'%key] = value
print('KPSS output for Temp is :')
print (kpss_output)

#first diferencing because the KPSS is not stationary
def f_difference(dataset, interval):
    diff = []

    for i in range(interval, len(dataset),interval):
        value = dataset[i] - dataset[i - interval]
        if i == 1:
            diff.append(0)
        elif i == 2 and interval == 2:
            diff.append(0)
            diff.append(0)
        elif i == 3 and interval == 3:
            diff.append(0)
            diff.append(0)
            diff.append(0)

        diff.append(value)
    return diff
x=f_difference(df['Temp'].values,1)

#Kpss after first differnecing
def kpss_test(x):
    print ('Results of KPSS Test:')
kpsstest = kpss(x, regression='c', nlags="auto")
kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','LagsUsed'])
for key,value in kpsstest[3].items():
    kpss_output['Critical Value (%s)'%key] = value
print('KPSS output for Temp is :')
print (kpss_output)

#Decomposition

Temp = df['Temp']
# df.replace(to_replace =["/"],
#             value ="-")
dates= pd.date_range(start='2013-02-20 11:40:00', periods=len(df),freq='1H')
temp_volume= pd.Series(df['Temp'].ravel(),index=dates)
STL = STL(temp_volume)
res = STL.fit()
fig = res.plot()
plt.xlabel("Time (Year)")
plt.suptitle('STL Decomposition', y=1.05)
plt.show()
#
# #seasonality and trend
T=res.trend
S=res.seasonal
R=res.resid
adj_seasonal=Temp - S
plt.plot(Temp,label='original')
plt.plot(adj_seasonal,label='Seasonality Adjusted')
plt.xlabel('time')
plt.ylabel('Temp')
plt.title('seasonality adjused data')
plt.legend()
plt.show()
#strength of tren
F=np.maximum(0,1-np.var(R)/np.var(np.array(T)+np.array(R)))
print(f'the strength of Trend is{F}')
#dtrended
T=res.trend
S=res.seasonal
R=res.resid
detrended=Temp - T
plt.plot(Temp,label='original')
plt.plot(detrended,label='detrended')
plt.xlabel('time')
plt.ylabel('Temp')
plt.title('adjusted detrended')
plt.legend()
plt.show()
#strength of seasonality
F=np.maximum(0,1-np.var(R)/np.var(np.array(S)+np.array(R)))
print(f'the strength of seasonality is{F}')

#holt_trend
n1=len(y_train)
n2 = len(y_test)
# train= df['Temp'][:n1]
# test= df['Temp'][n1:]
model=ets.ExponentialSmoothing(y_train, trend='add', damped_trend=True, seasonal=None).fit()
fitted= model.fittedvalues
forecast_holt= model.forecast(steps=len(y_test))
residual_error= y_train[1:].values-fitted[:-1].values
ACF_PACF_Plot(residual_error,20)
plt.hist(residual_error)
plt.show()
Q_holt=sm.stats.acorr_ljungbox(residual_error, lags=[20],return_df=True)
print(Q_holt)
plt.plot(list(range(0,n1)),y_train,label='Train')
plt.plot(list(range(n1,n1+n2)),y_test,label='Test')
plt.plot(list(range(n1,n1+n2)),forecast_holt,label='Forecast')
plt.title('forcast function for holt_trend')
plt.legend()
plt.show()
forcast_error_holt=y_test[2:].values-forecast_holt[:-2].values
#when the p value for Q is more than 0.05 is white which is not here now
#Chi square test
from scipy.stats import chi2
def chi_test(na,nb,lags,Q,e):
    DOF=lags-na-nb
    alpha=0.01
    chi_critical=chi2.ppf(1-alpha,DOF)
    if Q<chi_critical:
        print('The residuals are white')
    else:
        print('The residual is not white')
    lbvalue,pvalue=sm.stats.acorr_ljungbox(e,lags=[lags])
    print('From acorr_ljungbox test')
    print(lbvalue)
    print(pvalue)
chi_test(1,0,20,1257.221097,residual_error)
print('the estimated variance error of residual for holt is :',np.var(residual_error))
print('the estimated variance error of forcast for holt is :',np.var(forcast_error_holt))
ratio_variance_holt=np.var(residual_error)/np.var(forcast_error_holt)
print('the ratio bet var of pred and forcast in holt-trend:',ratio_variance_holt)

#MSE holt_tren for pred
mse_holt_trend_pred= metrics.mean_squared_error(fitted[1:],y_train[:-1])
print( "The Mean Square Error of predict for holt_trend is: " , (mse_holt_trend_pred))

#MSE holt_tren for forcast
mse_holt_trend_forcast= metrics.mean_squared_error(forecast_holt[2:],y_test[:-2])
print( "The Mean Square Error of forcast for holt-trend method is: " , (mse_holt_trend_forcast))

#Feature Selection
x=df.drop(['Temp','Site','Date','Time'],axis=1)
y=df['Temp']
x = sm.add_constant(x)
y=y.values
model=sm.OLS(y,x).fit()
print(model.summary())
x=x.drop(columns='const')
model2=sm.OLS(y,x).fit()
print(model2.summary())
#we remove contant because it's high std error
print('This is the final model',model2.summary())

#Average Method
#Average method and predict
t1= list(range(len(x_train))) #x_train
t2= list(range(len(x_train),len(x_test))) #x_test
n1=len(y_train)
n2 = len(y_test)
yhath_Average=[]
yhath_Average_val = np.mean(y_test)
#Prediction
for i in range(1,n1+1):
    yhath_Average.append(yhath_Average_val)
plt.plot(t1, y_train, color='green', marker='o', linestyle='dashed',  linewidth=2, markersize=12, label='prediction')
# plt.plot(t2, y_test, color='red', marker='x', linestyle='dashed',  linewidth=2, markersize=12,label='forcast')
plt.plot(t1, yhath_Average, color='blue', marker='o', linestyle='dashed',  linewidth=2, markersize=12, label='1-step')
plt.legend()
plt.xlabel('x')
plt.ylabel('yhat')
plt.title('average prediction method')
plt.show()
#calculating error for the average method
mse_Average_pred= metrics.mean_squared_error(yhath_Average[1:],y_train[:-1])
print( "The Mean Square Error of predict for Average method is: " , (mse_Average_pred))

#forcast
forcast=[]
for i in range(n2):
    forcast.append(yhath_Average[-1])
forcast=np.array(forcast)

train= df['Temp'][:n1]
test= df['Temp'][n1:]
plt.plot(list(range(0,n1)),train,label='Train')
plt.plot(list(range(n1,n1+n2)),test,label='Test')
plt.plot(list(range(n1,n1+n2)),forcast,label='Forecast')
plt.legend()
plt.title('average method forcast')
plt.show()
#calculating error for the average method forcast
mse_Average_forcast= metrics.mean_squared_error(forcast[2:],y_test[:-2])
print( "The Mean Square Error of forcast for Average method is: " , (mse_Average_forcast))

#Q value for average
residual_error_average=y_train[1:].values-yhath_Average[:-1]
ACF_PACF_Plot(residual_error_average,20)
Q=sm.stats.acorr_ljungbox(residual_error_average, lags=[100],return_df=True)
print(Q)
#forcast error for average
forcast_error_average=y_test[2:].values-forcast[:-2]
#variance of residual and forcast and the rate
print('the estimated variance error of residual for average is :',np.var(residual_error_average))
print('the estimated variance error of forcast for average is :',np.var(forcast_error_average))
ratio_variance_average=np.var(residual_error_average)/np.var(forcast_error_average)
print('the ratio bet var of pred and forcast in average:',ratio_variance_average)

#Naïve Method
pred_naive=[]
for T in range(len(train)):
    pred_naive.append(train.ravel()[T])
pred_naive=np.array(pred_naive)
#Plot NAive_prediction

t1= list(range(len(x_train))) #x_train
t2= list(range(len(y_test))) #x_test
plt.plot(t1, train, color='green', marker='o',  linewidth=2, markersize=12, label='train')
plt.plot(t1, pred_naive, color='blue', marker='o', linestyle='dashed',  linewidth=2, markersize=12, label='h-step')
plt.legend()
plt.xlabel('t')
plt.ylabel('yhat_naive')
plt.title('Naive Method')
plt.show()
#error for naive_prediction
mse_Naive= metrics.mean_squared_error(train[1:],pred_naive[:-1])
print( "The Mean Square Error of predict for Naive method is: " , (mse_Naive))

#forecast
forcast_naive=[]
for i in range(n2):
    forcast_naive.append(pred_naive.ravel()[-1])
forcast=np.array(forcast_naive)

train= df['Temp'][:n1]
test= df['Temp'][n1:]
plt.plot(list(range(0,n1)),train,label='Train')
plt.plot(list(range(n1,n1+n2)),test,label='Test')
plt.plot(list(range(n1,n1+n2)),forcast,label='Forecast')
plt.title('forcast for Naive')
plt.legend()
plt.show()
#mse error for naive_Forcast
mse_Naive_forcast= metrics.mean_squared_error(test[2:],forcast_naive[:-2])
print( "The Mean Square Error of predict for Naive method is: " , (mse_Naive_forcast))

#Q value for Naive
residual_error_naive=y_train[1:].values-pred_naive[:-1]
ACF_PACF_Plot(residual_error_naive,20)
Q_naive=sm.stats.acorr_ljungbox(residual_error_naive, lags=[100],return_df=True)
print(Q_naive)
chi_test(1,0,20,2.084765e+06,residual_error)
#forcast error for naive
forcast_error_naive=y_test[2:].values-forcast[:-2]
#variance of residual and forcast and the rate
print('the estimated variance error of residual for naive is :',np.var(residual_error_naive))
print('the estimated variance error of forcast for naive is :',np.var(forcast_error_naive))
ratio_variance_naive=np.var(residual_error_naive)/np.var(forcast_error_naive)
print('the ratio bet var of pred and forcast in naive:',ratio_variance_naive)

#Drift Method
h=1
drift_prediction=[]
for i in range (len(y_train.values)):
    drift_prediction.append(y_train.values[i]+h*(y_train.values[i]-y_train.values[0])/i)
drift_prediction=drift_prediction[1:-1]
plt.plot(y_train.values, color='green', label='original')
plt.plot(np.array(drift_prediction), color='blue',label='prediction')
plt.legend()
plt.xlabel('prediction')
plt.ylabel('yhat')
plt.title('drift prediction method')
plt.show()

# mse error for drift
mse_Drift= metrics.mean_squared_error(y_train[2:],drift_prediction)
print( "The Mean Square Error of predict for drift method is: " , (mse_Drift))

#Q value for drift
residual_error_drift=y_train[2:].values-drift_prediction
ACF_PACF_Plot(residual_error_drift,20)
Q_drift=sm.stats.acorr_ljungbox(residual_error_drift, lags=[100],return_df=True)
print(Q_drift)
chi_test(1,0,20,27782.818899,residual_error_drift)

#forcast for drift
drift_forcast=[]
for i in range (len(y_test.values)):
    drift_forcast.append(y_train.values[i]+h*(y_train.values[-1]-y_train.values[0])/len(y_train-1))
    h+=1

plt.plot(list(range(0,n1)),train.values,label='Train')
plt.plot(list(range(n1,n1+n2)),test.values,label='Test')
plt.plot(list(range(n1,n1+n2)),np.array(drift_forcast),label='Forecast')

plt.legend()
plt.xlabel('Forcast')
plt.ylabel('yhat')
plt.title('drift forcast method')
plt.show()

# mse error for drift of forcast
mse_Drift_forcast= metrics.mean_squared_error(y_test,drift_forcast)
print( "The Mean Square Error of forcast for drift method is: " , (mse_Drift))

#forcast error for drift
forcast_error_drift=y_test[2:].values-drift_forcast[:-2]
#variance of residual and forcast and the rate
print('the estimated variance error of residual for drift is :',np.var(residual_error_drift))
print('the estimated variance error of forcast for drift is :',np.var(forcast_error_drift))
ratio_variance_drift=np.var(residual_error_drift)/np.var(forcast_error_drift)
print('the ratio bet var of pred and forcast in drift:',ratio_variance_drift)


#SES predict
predict_SES=[train.tolist()[0]]
alfa=0.4
for i in range(len(train.values)):
    predict_SES.append(alfa*train.values[i]+(1-alfa)*predict_SES[i-1])

plt.plot(train.values, color='green', label='original')
plt.plot(np.array(predict_SES), color='blue',label='prediction')
plt.legend()
plt.xlabel('prediction')
plt.ylabel('yhat')
plt.title('SES prediction method')
plt.show()

#error for SES for predection
mse_SES= metrics.mean_squared_error(train[1:],predict_SES[1:-1])
print( "The Mean Square Error of predict for SES method is: " , (mse_SES))


#Q value for SES
residual_error_SES=train[1:].values-predict_SES[1:-1]
ACF_PACF_Plot(residual_error_SES,20)
Q_SES=sm.stats.acorr_ljungbox(residual_error_SES, lags=[100],return_df=True)
print(Q_SES)
chi_test(1,0,20,2.141921e+06,residual_error_SES)

#SES Forcast
SES_forcast=[]
for i in range(len(test)):
  SES_forcast.append(predict_SES[-1])

plt.plot(list(range(0,n1)),train.values,label='Train')
plt.plot(list(range(n1,n1+n2)),test.values,label='Test')
plt.plot(list(range(n1,n1+n2)),np.array(SES_forcast),label='Forecast')
plt.legend()
plt.xlabel('Forcast')
plt.ylabel('yhat')
plt.title('SES forcast method')
plt.show()

# mse error for drift of forcast
mse_SES_forcast= metrics.mean_squared_error(y_test,SES_forcast)
print( "The Mean Square Error of forcast for SES method is: " , (mse_SES_forcast))

#forcast error for SES
forcast_error_SES=y_test[2:].values-SES_forcast[:-2]
#variance of residual and forcast and the rate
print('the estimated variance error of residual for SES is :',np.var(residual_error_SES))
print('the estimated variance error of forcast for SES is :',np.var(forcast_error_SES))
ratio_variance_SES=np.var(residual_error_SES)/np.var(forcast_error_SES)
print('the ratio bet var of pred and forcast in SES:',ratio_variance_SES)


#Multiple linear Regression
X=df[['Latitude','Longitude','Depth']]
Y=df['Temp'].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, shuffle= False, test_size=0.2)

H=np.matmul(X_train.values.T,X_train.values)
print(H)
s,d,v = np.linalg.svd(H)
print('SingularValues =',d)

Condition_no=LA.cond(X_train)
print(f'Condition Number for train set: {Condition_no}')

model=sm.OLS(Y_train,X_train).fit()
print(model.summary())


#prediction of multiple linear regression
prediction=model.predict(X_train)
import matplotlib.pyplot as plt
plt.plot(list(range(len(Y_train))),Y_train,label='train')
# plt.plot(list(range(len(y_train),len(y_test))),Y_test,label='test_price')
plt.plot(list(range(len(Y_train))),prediction,label='pred')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Prediction for the multiple linear regression')
plt.show()
#Forcast_linear model
#Forcast of multiple linear regression
forecast=model.predict(X_test)
import matplotlib.pyplot as plt
plt.plot(list(range(len(y_train))),Y_train,label='train')
plt.plot(list(range(len(y_train),len(y_train)+len(y_test))),Y_test,label='test')
plt.plot(list(range(len(y_train),len(y_train)+len(y_test))),forecast,label='forecast')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Forcast for the linear regression')
plt.show()
#error MSE
MSE=np.square(np.subtract(Y_train[1:],prediction[:-1])).mean()
print('the Mean square error is:',MSE)
residual_error_linear=Y_train[1:]-prediction[:-1]
Q_linear=sm.stats.acorr_ljungbox(residual_error_linear, lags=[100],return_df=True)
print(Q_linear)
def autocorrelation(y, k):
    y_mean = np.mean(y)
    T = len(y)
    resultnumber = 0
    res_den = 0
    for t in range(k,T):
        resultnumber += (y[t] - y_mean) * (y[t-k] - y_mean)

    for t in range(0,T):
        res_den += (y[t] - y_mean)*(y[t] - y_mean)

    res = resultnumber/res_den
    return res

def auto_corr_cal(y, k):
    res = []
    for t in range(0, k):
        result = autocorrelation(y, t)
        res.append(result)
    return res

##forcast error for linear regression
Forcast_error=Y_test-forecast
print(Forcast_error)
Q_linear_forcast=sm.stats.acorr_ljungbox(Forcast_error, lags=[100],return_df=True)
print(Q_linear_forcast)
#error for the forcast of linear

MSE_linear_forcast=np.square(np.subtract(Y_test[1:],forecast.values[:-1])).mean()
print('the Mean square error is:',MSE_linear_forcast)

ACF_3_Forecast=auto_corr_cal(Forcast_error.values, 20)
ry= ACF_3_Forecast[::-1][:-1] + ACF_3_Forecast
ac= ACF_3_Forecast
plt.stem(np.linspace(-((len(ac))-1), len(ac)-1, (len(ac)*2-1), dtype=int), ry, markerfmt='o')
plt.axhspan(-(1.96/np.sqrt(len(Y_test))), (1.96/np.sqrt(len(Y_test))), alpha=0.2, color='blue')
plt.title('the ACF for the Forcast in linear regression')
plt.show()

#estimated variance
def estimated_variance(T,K,E):#K number of factors # T number of total samples
    a=1/(T-K-1)
    sum_e=0
    for i in range(1,T):
        sum_e=sum_e+(E[i]**2)
    e_v=np.square(a*sum_e)
    return e_v
print('the estimated variance error of residual for linea is :',np.var(residual_error_linear))
print('the estimated variance error of forcast for linear is :',np.var(Forcast_error.values))
ratio_variance_linear=np.var(residual_error_linear)/np.var(Forcast_error.values)
print('the ratio bet var of pred and forcast in linear:',ratio_variance_linear)

#T_Test
print(model.summary())
A=np.identity(len(model.params))
print('f test result is',model.f_test(A))
print('t test for the model is',model.t_test(A))

#ACF Function
def autocorrelation(y, k):
    y_mean = np.mean(y)
    T = len(y)
    resultnumber = 0
    res_den = 0
    for t in range(k,T):
        resultnumber += (y[t] - y_mean) * (y[t-k] - y_mean)

    for t in range(0,T):
        res_den += (y[t] - y_mean)*(y[t] - y_mean)

    res = resultnumber/res_den
    return res

def auto_corr_cal(y, k):
    res = []
    for t in range(0, k):
        result = autocorrelation(y, t)
        res.append(result)
    return res
ACF_for_raw=auto_corr_cal(y,100)
ac= ACF_for_raw
acc=ac[::-1]+ac[1:]

#GPAC
#den and NUm
def phi_kk(j,k,ry):
    num = [];a =[];b = []
    den = []
    M = math.ceil(len(ry)/2)
    if k==1:
        phi_kk = ry[M+j]/ry[M+j-1]
    else:
        for m in range(k):

            den.append(ry[M + j + m-1: M + j + m - k-1: -1])
            b.append(ry[M + j+ m])

        den = np.array(den)
        x = np.array(b)
        c = np.transpose([x])
        a = den[...,0:k-1]
        num = np.concatenate((a,c),axis=1)

        if np.linalg.det(den)!=0:
            phi_kk = np.linalg.det(num)/np.linalg.det(den)
        else:
            phi_kk = 'inf'
    return phi_kk
def pm(s):
    j,k = s.shape
    GPAC = pd.DataFrame(s)
    GPAC.columns = range(1, k+1)
    return GPAC

def Cal_GPAC(ryy, j, k):
    phi = np.zeros((j,k-1))
    for kk in range(1,k):
        for jj in range(j):
            phi_kk1= phi_kk(jj,kk,ryy)
            phi[jj][kk-1]=phi_kk1
    return pm(phi)
gpac_table= Cal_GPAC(acc,10,10)
sns.heatmap(gpac_table,annot=True)
plt.title('GPAC')
plt.show()
gpac_table= Cal_GPAC(acc,5,5)
sns.heatmap(gpac_table,annot=True)
plt.title('GPAC')
plt.show()

#The pattern is 1 and 0 for the GPAc
#ARMA
na=1 #AR auto correlation
nb=0 #moving Average
y=train

model=sm.tsa.ARMA(y,(na,nb)).fit(trend='nc',disp=0)

for i in range(na):
    print('The AR coefficient a{}'.format(i),'is:',model.params[i])
for i in range(nb):
    print('The MA coefficient a{}'.format(i),'is:',model.params[i+na])
print(model.summary())

#Prediction for ARMA
arma10_predict=model.predict(start=0,end=len(train)-1)
arma10_res=train.values[1:]-arma10_predict[:-1]
#Q for ARMA
Q_ARMA_pred=sm.stats.acorr_ljungbox(arma10_res, lags=[100],return_df=True)
print(Q_ARMA_pred)
#Forecast
arma10_test=model.predict(start=len(train), end=len(df))
arma10_fore_error=test.values[2:]-arma10_test[:-3]

#Chi square test
chi_test(1,0,20,30458.815462,arma10_res)
ACF_PACF_Plot(arma10_res,20)

#mse for ARMA for predict
mse_ARMA_predict= metrics.mean_squared_error(train[1:],arma10_predict[:-1])
print( "The Mean Square Error of predict for ARMA is: " , (mse_ARMA_predict))

#predict ARMA
plt.plot(train.values, color='green', label='original')
plt.plot(np.array(arma10_predict), color='blue',label='prediction')
plt.legend()
plt.xlabel('prediction')
plt.ylabel('yhat')
plt.title('ARMA prediction method')
plt.show()
##ARMA forcast
n1=len(y_train)
n2 = len(y_test)
train= df['Temp'][:n1]
test= df['Temp'][n1:]
plt.plot(list(range(n1)),train.values,label='Train')
plt.plot(list(range(n1,n1+n2)),test.values,label='Test')
plt.plot(list(range(n1,n1+n2)),np.array(arma10_test)[:-1],label='Forecast')
plt.legend()
plt.xlabel('Forcast')
plt.ylabel('yhat')
plt.title('Arma forcast method')
plt.show()
#MSE forcast for ARMA
mse_ARMA_forcast= metrics.mean_squared_error(test[2:],arma10_test[:-3])
print( "The Mean Square Error of forcast for ARMA is: " , (mse_ARMA_forcast))

#estimated variance of error for ARMA oredict and Forcast
print('the variance of residual error',estimated_variance(5,160,np.array(arma10_res)))
print('the variance of forcast error',estimated_variance(5,160,arma10_fore_error.values))

#diagnostic test
#chi2 test
chi_test(1,0,20,30458.815462,arma10_res)

#Confidence interval #need to work on that
for i in range(na+nb):
    print(f'Confidence interval:\n{model.params[i]}: {model.conf_int()[i]}')

ratio_residual_forcast=estimated_variance(5,160,np.array(arma10_res))/estimated_variance(5,160,arma10_fore_error.values)
print('ratio residual forcast',ratio_residual_forcast)
#it is a biased model because bet the diffrence estimated parameters are not zero

