import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

df = pd.DataFrame({"Fruit": ["Apples", "Oranges", "Bananas",
                             "Apples", "Oranges", "Bananas"],
                   "Amount": [4, 1, 2, 2, 4, 5],
                   "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]})

# a plotly figure
fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

# our web app
app.layout = html.Div(children=[html.H1(children='This is a title'),
                                html.Div(children='This is my web app in Dash'),
                                dcc.Graph(figure=fig)])

if __name__ == '__main__':
    app.run_server(debug=True)
