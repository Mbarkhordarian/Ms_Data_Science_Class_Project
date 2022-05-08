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

#Dashbord
external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']
my_app = dash.Dash('My app',external_stylesheets=external_stylesheets)

my_app.layout=html.Div([
    html.H1('Final Project',style={'textAlign':'center'}),
    html.Br(),
    dcc.Tabs(id='hw-questions',
             children=[
                 dcc.Tab(label='Visualization1',value='Visualization1'),
                 dcc.Tab(label='Visualization2',value='Visualization2'),
                 dcc.Tab(label = 'download', value = 'download'),
                 dcc.Tab(label = 'statistic', value = 'statistic')
             ]),
    html.Div(id='layout')
])

Visualization_layout=html.Div([
    html.H1('Please pick the feature'),
    dcc.Dropdown(id = 'Status',
                 options=[
                     {'label':'Base Pay', 'value':'Base Pay'},
                     {'label':'Overtime Pay', 'value':'Overtime Pay'},

                 ], value='Base Pay', clearable=False),
    html.H2('Please pick the output variable'),
    dcc.Dropdown(id = 'Year',
                 options=[
                     {'label':'Year', 'value':'Year'},
                     {'label':'Status', 'value':'Status'},

                 ], value='Year', clearable=False),
    html.H4('Line Plot'),
    dcc.Graph(id='my-graph'),
    html.Br(),
    html.H1('Please pick the feature'),
    dcc.Dropdown(id='Benefits',options=[
        {'label':'Overtime Pay', 'value':'Overtime Pay'},
        {'label':'Base Pay', 'value':'Base Pay'},
        {'label':'Other Pay', 'value':'Other Pay'},
        {'label':'Benefits', 'value':'Benefits'},
        {'label':'Total Pay', 'value':'Total Pay'},
        {'label':'Total Pay & Benefits', 'value':'Total Pay & Benefits'}

    ], value='Overtime Pay', clearable=False),
    html.H2('Please pick the output variable'),
    dcc.RadioItems(['Year'],value='Year',id='date'),
    html.H4('Violin Plot'),
    dcc.Graph(id='my-graph2')

])
Visualization2_layout=html.Div([
    html.H1('Please pick the feature'),
    dcc.Dropdown(id='Benefits',options=[
        {'label':'Overtime Pay', 'value':'Overtime Pay'},
        {'label':'Base Pay', 'value':'Base Pay'},
        {'label':'Other Pay', 'value':'Other Pay'},
        {'label':'Benefits', 'value':'Benefits'},
        {'label':'Total Pay', 'value':'Total Pay'},
        {'label':'Total Pay & Benefits', 'value':'Total Pay & Benefits'}

         ], value='Overtime Pay', clearable=False),
    html.H2('Please pick the output variable'),
    dcc.RadioItems(['Status'],value='Status',id='Status'),
    html.H4('Histogram Plot with violin marginal'),
    dcc.Graph(id='my-graph3'),
    html.Br(),
    html.Br(),
    html.H1('Please pick the feature'),
    dcc.Dropdown(id='Benefits2',options=[
        {'label':'Overtime Pay', 'value':'Overtime Pay'},
        {'label':'Base Pay', 'value':'Base Pay'},
        {'label':'Other Pay', 'value':'Other Pay'},
        {'label':'Benefits', 'value':'Benefits'},
        {'label':'Total Pay', 'value':'Total Pay'},
        {'label':'Total Pay & Benefits', 'value':'Total Pay & Benefits'}

    ], value='Overtime Pay', clearable=False),
    html.H2('Please pick the output variable'),
    dcc.Checklist(['Status'],value='Status',id='Status'),
    html.H4('Histogram Plot with box marginal'),
    dcc.Graph(id='my-graph4'),
])

download_layout=html.Div([
    html.H1('Download the cleaned dataset', style={'textAlign':'center'}),
    html.Br(),
    html.Label('Click button to download csv file'),
    html.Br(),
    html.Button(id='download1', children='Download'),
    dcc.Download(id='download2')
])
#call back for download
@my_app.callback(Output(component_id="download2", component_property="data"),
              Input(component_id="download1", component_property="n_clicks"),
              prevent_initial_call=True)

def displaydown_layout(sel1):
    return dcc.send_data_frame(df.to_csv, "SF_payroll.csv")

statistic_layout=html.Div([
    html.H1('Basic statistics', style={'textAlign':'center'}),
    html.Br(),
    html.Br(),
    html.P('Radiobox to select Variable'),
    html.Br(),
    dcc.RadioItems(
        id='radio',
        options=[
            {'label':'Overtime Pay', 'value':'Overtime Pay'},
            {'label':'Base Pay', 'value':'Base Pay'},
            {'label':'Other Pay', 'value':'Other Pay'},
            {'label':'Benefits', 'value':'Benefits'},
            {'label':'Total Pay', 'value':'Total Pay'},
            {'label':'Total Pay & Benefits', 'value':'Total Pay & Benefits'},
        ],value=['Overtime Pay'],
    ),
    html.Br(),
    html.Br(),
    html.H2(id='output1'),
    html.Br(),
    html.H2(id='output2'),
    html.Br(),
    html.H2(id='output3'),
    html.Br(),
    html.H2(id='output4'),
    html.Br(),
    html.H2(id='output5')
])

@my_app.callback(
    [Output(component_id='output1', component_property='children'),
     Output(component_id='output2', component_property='children'),
     Output(component_id='output3', component_property='children'),
     Output(component_id='output4', component_property='children'),
     Output(component_id='output5', component_property='children'),],
    [Input(component_id='radio',component_property='value'),]
)

def display_color(sel1):

        sent = f' This Variable is Numeric'
        mean_df = f'mean : {np.mean(df[sel1])}'
        median_df = f'median : {np.median(df[sel1])}'
        std_df = f'standard deviation : {np.std(df[sel1])}'
        var_df = f'Variance : {np.var(df[sel1])}'

        return sent, mean_df, median_df, std_df, var_df

#first tab1-graph lay out
@my_app.callback(
    Output(component_id='my-graph', component_property='figure'),
    [Input(component_id='Status', component_property='value'),
     Input(component_id='Year', component_property='value'),]
)
def undate_n(Status, Year):
    fig=px.line(df,x=Year,y=Status)
    return fig
#second graph call back
@my_app.callback(
    Output(component_id='my-graph2', component_property='figure'),
    [Input(component_id='Benefits', component_property='value'),
     Input(component_id='date', component_property='value')]
)
def undate_n2(Benefits, date):

    #fig = px.pie(df,values=Year, names=Status)
    fig=px.violin(df,x=date,y=Benefits)
    return fig


#graph3 in tab2
@my_app.callback(
    Output(component_id='my-graph3', component_property='figure'),
    [Input(component_id='Status', component_property='value'),
     Input(component_id='Benefits', component_property='value'),]
)
def undate_3(Status, Benefits):

    fig = px.histogram(df, x=Benefits,color=Status,
                       nbins=50, marginal = 'violin')
    return fig

#graph4 in tab2
@my_app.callback(
    Output(component_id='my-graph4', component_property='figure'),
    [Input(component_id='Status', component_property='value'),
     Input(component_id='Benefits2', component_property='value'),]
)
def undate_4(Status, Benefits):

    fig = px.histogram(df, x=Benefits,color=Status,
                       nbins=50, marginal = 'box')
    return fig


#tabs
@my_app.callback(Output(component_id='layout',component_property='children'),

                 [Input(component_id='hw-questions',component_property='value'),
                  ])

def update_layout(ques):
    if ques=='Visualization1':
        return Visualization_layout
    elif ques=='Visualization2':
        return Visualization2_layout
    elif ques=='download':
        return download_layout
    elif ques=='statistic':
        return statistic_layout

my_app.run_server(port=2626,
                  host='0.0.0.0')
