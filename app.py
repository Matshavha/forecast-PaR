import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
import plotly.express as px
import holidays

# Set directories to the current directory since models, Excel files, and app are in the same folder
data_directory = "Files"
model_directory = "Files"

dataset_names = [
    "Megaflex_Hourly", "Miniflex_Hourly", "Ruraflex_Hourly", "Transflex_Hourly",
    "Megaflex_Munic_Hourly", "Miniflex_Munic_Hourly", "Ruraflex_Munic_Hourly",
    "Nightsave_Urban_Hourly", "Nightsave_Rural_Hourly", "Nightsave_Urban_Munic_Hourly", "Nightsave_Rural_Munic_Hourly"
]

def load_model(dataset_name, model_type):
    model_map = {
        "Total Consumption (kWh)": "consumption",
        "Apparent Power (kVA)": "apparent",
        "Reactive Power": "reactive"
    }
    model_file = f"{dataset_name}_{model_map[model_type]}.pkl"
    model_path = os.path.join(model_directory, model_file)  # Now loads from 'Files' folder
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

def load_data(dataset_name):
    file_path = os.path.join(data_directory, f"{dataset_name}_last_170_hours.xlsx")
    return pd.read_excel(file_path)


def preprocess_features(df):
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values(by='DateTime')
    df['year'] = df['DateTime'].dt.year
    df['month'] = df['DateTime'].dt.month
    df['day'] = df['DateTime'].dt.day
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Add South African holidays
    sa_holidays = holidays.ZA(years=df['DateTime'].dt.year.unique())
    df['is_holiday'] = df['DateTime'].dt.date.apply(lambda x: int(x in sa_holidays))
    
    # Compute rolling means
    df['rolling_mean_6'] = df['Total Consumption (kWh)'].rolling(window=6).mean()
    df['rolling_mean_12'] = df['Total Consumption (kWh)'].rolling(window=12).mean()
    
    return df.dropna()

def create_future_features(df, forecast_period):
    future_dates = pd.date_range(start=df['DateTime'].max(), periods=forecast_period + 1, freq='H')[1:]
    future_df = pd.DataFrame({'DateTime': future_dates})

    # Generate future time-based features
    future_df['year'] = future_df['DateTime'].dt.year
    future_df['month'] = future_df['DateTime'].dt.month
    future_df['day'] = future_df['DateTime'].dt.day
    future_df['hour'] = future_df['DateTime'].dt.hour
    future_df['day_of_week'] = future_df['DateTime'].dt.dayofweek
    future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)

    # Ensure 'is_holiday' feature exists
    sa_holidays = holidays.ZA(years=future_df['year'].unique())
    future_df['is_holiday'] = future_df['DateTime'].dt.date.apply(lambda x: int(x in sa_holidays))

    # Fill rolling means with last known value
    future_df['rolling_mean_6'] = df['rolling_mean_6'].iloc[-1] if 'rolling_mean_6' in df.columns else df['Total Consumption (kWh)'].rolling(6).mean().iloc[-1]
    future_df['rolling_mean_12'] = df['rolling_mean_12'].iloc[-1] if 'rolling_mean_12' in df.columns else df['Total Consumption (kWh)'].rolling(12).mean().iloc[-1]

    # Carry forward lagged values only if enough historical data is available
    for lag in [1, 2, 3, 6, 24, 168]:
        if len(df) >= lag:
            future_df[f'lag_{lag}'] = df['Total Consumption (kWh)'].iloc[-lag]
            future_df[f'lag_{lag}_apparent'] = df['Apparent Power (kVA)'].iloc[-lag]
            if 'Reactive Power' in df.columns:
                future_df[f'lag_{lag}_reactive'] = df['Reactive Power'].iloc[-lag]
        else:
            future_df[f'lag_{lag}'] = df['Total Consumption (kWh)'].iloc[-1]
            future_df[f'lag_{lag}_apparent'] = df['Apparent Power (kVA)'].iloc[-1]
            if 'Reactive Power' in df.columns:
                future_df[f'lag_{lag}_reactive'] = df['Reactive Power'].iloc[-1]

    return future_df

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.layout = dbc.Container([
    html.Div(
        style={
            'display': 'flex',
            'justify-content': 'center',  # Centers content horizontally
            'align-items': 'center',  # Centers content vertically
            'gap': '15px',  # Space between logo and title
            'margin-bottom': '10px'
        },
        children=[
            html.Img(
                src="/assets/logo.jpg",  # Ensure you place your logo file in the 'assets' folder
                style={'height': '100px', 'width': 'auto'}
            ),
            html.H1("Energy Forecasting Dashboard", className="text-center", style={'margin': '0'})
        ]
    ),

    html.Div(
        style={'display': 'flex', 'flex-wrap': 'wrap', 'align-items': 'center', 'gap': '10px'},
        children=[
            html.Div(
                style={'flex-grow': '1', 'min-width': '250px'},
                children=[
                    html.Label("Select Dataset", style={'font-weight': 'bold'}),
                    dcc.Dropdown(
                        id='dataset-dropdown',
                        options=[{'label': name, 'value': name} for name in dataset_names] + [{'label': "All Tariffs", 'value': "All Tariffs"}],
                        value=dataset_names[0],
                        clearable=False,
                        style={'width': '50%'}
                    )
                ]
            ),

            html.Div(
                style={'min-width': '180px'},
                children=[
                    html.Button("Download Forecast", id="download-button", className="btn btn-success", style={'width': '100%', 'height': '40px'})
                ]
            ),

            dcc.Download(id="download-forecast-data"),  # Hidden download component for CSV download
            
            # Power BI Button & Container
            html.Div(
                style={'min-width': '180px'},
                children=[
                    html.Button("View Power BI Report", id="powerbi-button", className="btn btn-info", style={'width': '100%', 'height': '40px'}),
                    html.Div(id="powerbi-container", style={'width': '100%', 'margin-top': '15px'}),
                ]
            ),
        ]
    ),

    dbc.Row([
        dbc.Col([
            html.Label("Select Forecast Period (Hours)", style={'font-weight': 'bold'}),
            dcc.Input(id='forecast-period', type='number', value=24, min=1, max=168, step=1, style={'width': '20%'})
        ], width=6, xs=12),
        dbc.Col([
            html.Button("Generate Forecast", id="predict-button", className="btn btn-primary mt-4", style={'width': '100%'})
        ], width=6, xs=12)
    ], className="mt-3"),

    html.Hr(),

    html.Div(
        style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'width': '100%'},
        children=[
            dcc.Graph(id='forecast-graph', style={'width': '100%', 'max-width': '900px', 'height': '400px'}),
            dcc.Graph(id='power-factor-graph', style={'width': '100%', 'max-width': '800px', 'height': '300px', 'margin-top': '10px'})
        ]
    ),

    html.Footer(
        children=[
            html.P("Developed by Dx - Electricity Pricing | © 2025 All Rights Reserved", style={'text-align': 'center'})
        ],
        style={'position': 'fixed', 'bottom': '0', 'width': '100%', 'background-color': '#f8f9fa', 'padding': '10px'}
    )
], fluid=True)

@app.callback(
    [Output('forecast-graph', 'figure'),
     Output('power-factor-graph', 'figure')],
    Input('predict-button', 'n_clicks'),
    State('dataset-dropdown', 'value'),
    State('forecast-period', 'value')
)
def update_forecast(n_clicks, dataset_name, forecast_period):
    if n_clicks is None:
        return px.line(), px.line()

    target_columns = ["Total Consumption (kWh)", "Apparent Power (kVA)", "Reactive Power"]
    forecast_results = {target: np.zeros(forecast_period) for target in target_columns}

    if dataset_name == "All Tariffs":
        for name in dataset_names[:-1]:
            df = load_data(name)
            df = preprocess_features(df)
            future_df = create_future_features(df, forecast_period)

            for target_column in target_columns:
                try:
                    model = load_model(name, target_column)
                    X = future_df.drop(columns=['DateTime'], errors='ignore')

                    expected_features = model.get_booster().feature_names
                    X = X[expected_features]

                    forecast_results[target_column] += model.predict(X)
                except FileNotFoundError:
                    continue
    else:
        df = load_data(dataset_name)
        df = preprocess_features(df)
        future_df = create_future_features(df, forecast_period)

        for target_column in target_columns:
            try:
                model = load_model(dataset_name, target_column)
                X = future_df.drop(columns=['DateTime'], errors='ignore')

                expected_features = model.get_booster().feature_names
                X = X[expected_features]

                forecast_results[target_column] = model.predict(X)
            except FileNotFoundError:
                continue

    # Scale down values to MWh/MVA/MVarh
    future_df['Consumption Forecast'] = forecast_results["Total Consumption (kWh)"] / 1000
    future_df['Apparent Power Forecast'] = forecast_results["Apparent Power (kVA)"] / 1000
    future_df['Reactive Power Forecast'] = forecast_results["Reactive Power"] / 1000

    # Compute Power Factor = Active Power (kW) / Apparent Power (kVA)
    future_df['Power Factor'] = np.where(
        future_df['Apparent Power Forecast'] == 0, 0, 
        future_df['Consumption Forecast'] / future_df['Apparent Power Forecast']
    )

    forecast_fig = px.line(
        future_df, x='DateTime', 
        y=["Consumption Forecast", "Apparent Power Forecast", "Reactive Power Forecast"],
        labels={"value": "Energy (MWh/MVA/MVarh)"},
        title=f"{dataset_name} - Future Forecast for {forecast_period} hours"
    )

    power_factor_fig = px.line(
        future_df, x='DateTime', y="Power Factor",
        labels={"Power Factor": "Power Factor"},
        title=f"{dataset_name} - Power Factor Forecast"
    )

    return forecast_fig, power_factor_fig
    
@app.callback(
    Output("download-forecast-data", "data"),
    Input("download-button", "n_clicks"),
    State("dataset-dropdown", "value"),
    State("forecast-period", "value"),
    prevent_initial_call=True
)
def download_forecast(n_clicks, dataset_name, forecast_period):
    combined_results = []

    if dataset_name == "All Tariffs":
        for name in dataset_names:
            df = load_data(name)  # Load dataset
            df = preprocess_features(df)  # Preprocess
            future_df = create_future_features(df, forecast_period)  # Create future features

            target_columns = ["Total Consumption (kWh)", "Apparent Power (kVA)", "Reactive Power"]
            forecast_results = {target: np.zeros(forecast_period) for target in target_columns}

            for target_column in target_columns:
                try:
                    model = load_model(name, target_column)  # Load model per tariff
                    X = future_df.drop(columns=['DateTime'], errors='ignore')
                    expected_features = model.get_booster().feature_names
                    X = X[expected_features]

                    forecast_results[target_column] = model.predict(X)  # Predict for each tariff
                except FileNotFoundError:
                    continue

            # Create forecast dataframe per tariff
            tariff_forecast = future_df[['DateTime']].copy()  # Keep only DateTime
            tariff_forecast['Consumption Forecast'] = forecast_results["Total Consumption (kWh)"] / 1000
            tariff_forecast['Apparent Power Forecast'] = forecast_results["Apparent Power (kVA)"] / 1000
            tariff_forecast['Reactive Power Forecast'] = forecast_results["Reactive Power"] / 1000
            tariff_forecast['Power Factor'] = np.where(
                tariff_forecast['Apparent Power Forecast'] == 0, 0, 
                tariff_forecast['Consumption Forecast'] / tariff_forecast['Apparent Power Forecast']
            )
            tariff_forecast["Tariff Plan"] = name.replace("_Hourly", "")  # Assign tariff plan

            combined_results.append(tariff_forecast)  # Store individual tariff results

        df = pd.concat(combined_results, ignore_index=True)  # Combine all tariffs

    else:
        df = load_data(dataset_name)
        df = preprocess_features(df)
        future_df = create_future_features(df, forecast_period)

        target_columns = ["Total Consumption (kWh)", "Apparent Power (kVA)", "Reactive Power"]
        forecast_results = {target: np.zeros(forecast_period) for target in target_columns}

        for target_column in target_columns:
            try:
                model = load_model(dataset_name, target_column)
                X = future_df.drop(columns=['DateTime'], errors='ignore')
                expected_features = model.get_booster().feature_names
                X = X[expected_features]

                forecast_results[target_column] = model.predict(X)
            except FileNotFoundError:
                continue

        future_df = future_df[['DateTime']].copy()  # Keep only DateTime
        future_df['Consumption Forecast'] = forecast_results["Total Consumption (kWh)"] / 1000
        future_df['Apparent Power Forecast'] = forecast_results["Apparent Power (kVA)"] / 1000
        future_df['Reactive Power Forecast'] = forecast_results["Reactive Power"] / 1000
        future_df['Power Factor'] = np.where(
            future_df['Apparent Power Forecast'] == 0, 0, 
            future_df['Consumption Forecast'] / future_df['Apparent Power Forecast']
        )
        future_df["Tariff Plan"] = dataset_name.replace("_Hourly", "")

        df = future_df  # Single dataset results

    # Ensure DateTime is sorted in order for each Tariff Plan
    df = df.sort_values(by=["Tariff Plan", "DateTime"]).reset_index(drop=True)

    # Download as CSV
    return dcc.send_data_frame(df.to_csv, filename=f"{dataset_name}_forecast_{forecast_period}h.csv", index=False)
@app.callback(
    Output("powerbi-container", "children"),
    Input("powerbi-button", "n_clicks"),
    prevent_initial_call=True
)
def display_powerbi_report(n_clicks):
    powerbi_embed_url = "https://app.powerbi.com/reportEmbed?reportId=a835a316-18ba-442b-bba0-a8a5967b9a44&autoAuth=true&ctid=93aedbdc-cc67-4652-aa12-d250a876ae79"

    return html.Div(
        children=[
            html.Iframe(
                src=powerbi_embed_url,
                style={
                    "position": "fixed",
                    "top": "0",
                    "left": "0",
                    "width": "100vw",  # Full viewport width
                    "height": "100vh",  # Full viewport height
                    "border": "none",
                    "z-index": "1000",  # Ensures it stays above all other elements
                    "background": "white"
                }
            )
        ],
        style={
            "position": "fixed",
            "top": "0",
            "left": "0",
            "width": "100vw",
            "height": "100vh",
            "z-index": "1000",
            "background": "white"
        }
    )

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8000, debug=True)


