import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Initialize the Dash app
app = dash.Dash(__name__)

# Sample data for visualization
df = pd.DataFrame({
    "Category": ["A", "B", "C", "D"],
    "Values": [10, 20, 30, 40]
})

# Create a simple bar chart
fig = px.bar(df, x="Category", y="Values", title="Sample Bar Chart")

# Define the layout of the app
app.layout = html.Div([
    html.H1("Basic Dash App"),
    dcc.Graph(figure=fig)
])

# Define the server for Azure
server = app.server

# Run the app locally
if __name__ == "__main__":
    app.run_server(debug=True)
