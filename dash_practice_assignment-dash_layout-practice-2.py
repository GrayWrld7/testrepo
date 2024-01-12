import pandas as pd 
import dash 
import dash_core_components as dcc 
import dash_html_components as html 
from dash.dependencies import Input, Output 
import plotly.graph_objects as go 
import plotly.express as px 

#airline_data = pd.read_csv('',
    #encoding='',
    #dtype={'':,
    #}
#)

# Add Dataframe
fruit_data = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "NYC", "MTL", "NYC"]
    })

# Add a bar graph figure
fig = px.bar(fruit_data,x="Fruit",y="Amount",color="City",barmode="group")

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(
        children='Dashboard',
        style={'textAlign':'center','font-size':40,'color':'#503D36'}
    ),
    html.Div(['Input Year: ',
        dcc.Input(id='input-year',
            value='2010',
            type='number',
            style={'height':'50px','font-size':30})],
            style={'font-size':35}),
    html.Br(),
    html.Br(),
    html.Div(dcc.Dropdown(options=[
        {'label':'New York','value':'NYC'},
        {'label':u'Montréal','value':'MTL'},
        {'label':'San Francisco','value':'SF'},
    ],
    value='NYC')
    ),
    html.Div(dcc.Graph(id='example-graph-2',figure=fig))
    
    ])


if __name__ == '__main__':
    app.run_server()