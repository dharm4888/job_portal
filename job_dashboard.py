import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # Added missing import
import numpy as np
from datetime import datetime
import traceback

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # For deployment

# Load and preprocess data with comprehensive error handling
try:
    df = pd.read_csv('cleaned_jobs.csv')
    print(f"Data loaded successfully with {len(df)} rows")
    
    # Ensure correct types and fill missing values
    if 'published_date' in df.columns:
        df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
    else:
        df['published_date'] = pd.date_range('2023-01-01', periods=len(df), freq='D')
    
    # Handle salary column
    if 'salary' in df.columns:
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
        # Remove unrealistic salaries
        df = df[(df['salary'] > 1000) & (df['salary'] < 1000000)]
        df['salary'].fillna(df['salary'].median(), inplace=True)
    else:
        df['salary'] = np.random.normal(70000, 20000, len(df)).astype(int)
    
    # Handle remote work column
    if 'is_remote' in df.columns:
        df['is_remote'] = df['is_remote'].fillna(False)
        # Convert to int (0/1)
        df['is_remote'] = df['is_remote'].astype(int)
    else:
        df['is_remote'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
    
    # Handle country column
    if 'country' in df.columns:
        df['country'].fillna('Unknown', inplace=True)
        # Only include countries with sufficient data
        country_counts = df['country'].value_counts()
        valid_countries = country_counts[country_counts > 5].index.tolist()
        df = df[df['country'].isin(valid_countries)]
    else:
        countries = ['USA', 'UK', 'Canada', 'Australia', 'Germany', 'India']
        df['country'] = np.random.choice(countries, len(df))
    
    # Create month_year column safely
    df['month_year'] = df['published_date'].dt.strftime('%Y-%m')
    df['month_year'].fillna('Unknown', inplace=True)
    
    # Ensure we have data for all months
    if len(df['month_year'].unique()) < 2:
        # Create dummy data if not enough months
        min_date = df['published_date'].min()
        max_date = df['published_date'].max()
        if pd.isna(min_date) or pd.isna(max_date):
            min_date = pd.Timestamp('2023-01-01')
            max_date = pd.Timestamp('2023-12-31')
        
        all_months = pd.date_range(min_date, max_date, freq='MS').strftime('%Y-%m').tolist()
        # Ensure we have at least 6 months of data
        if len(all_months) < 6:
            all_months = pd.date_range('2023-01-01', '2023-12-31', freq='MS').strftime('%Y-%m').tolist()
    
    print(f"Data processed successfully. Countries: {df['country'].unique()}")
    
except Exception as e:
    print(f"Error loading data: {e}")
    print(traceback.format_exc())
    # Create sample data for demonstration if file not found
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    countries = ['USA', 'UK', 'Canada', 'Australia', 'Germany', 'India']
    
    df = pd.DataFrame({
        'published_date': np.random.choice(dates, 1000),
        'salary': np.random.normal(70000, 20000, 1000).astype(int),
        'is_remote': np.random.choice([0, 1], 1000, p=[0.7, 0.3]),
        'country': np.random.choice(countries, 1000),
        'job_title': np.random.choice(['Developer', 'Analyst', 'Manager', 'Designer'], 1000)
    })
    df['published_date'] = pd.to_datetime(df['published_date'])
    df['month_year'] = df['published_date'].dt.strftime('%Y-%m')

# Get unique countries for dropdown
country_options = [{'label': 'All Countries', 'value': 'ALL'}]
for country in sorted(df['country'].unique()):
    country_options.append({'label': country, 'value': country})

# Layout
app.layout = html.Div([
    html.H1("Job Market Dynamics Dashboard", style={'textAlign': 'center'}),
    
    html.Div([
        dcc.Dropdown(
            id='metric',
            options=[
                {'label': 'Total Postings', 'value': 'count'},
                {'label': 'Average Salary', 'value': 'salary'},
                {'label': 'Remote Work Percentage', 'value': 'remote_percentage'},
                {'label': 'Average Salary by Country', 'value': 'salary_by_country'}
            ],
            value='count',
            style={'width': '50%', 'margin': '10px auto'}
        ),
        
        dcc.Dropdown(
            id='country-selector',
            options=country_options,
            value='ALL',
            style={'width': '50%', 'margin': '10px auto'},
            disabled=False
        ),
    ], style={'padding': '20px'}),
    
    dcc.Graph(id='trend-graph'),
    
    html.Div([
        dcc.RangeSlider(
            id='salary-range',
            min=int(df['salary'].min()),
            max=int(df['salary'].max()),
            step=10000,
            value=[int(df['salary'].quantile(0.25)), int(df['salary'].quantile(0.75))],
            marks={i: f'${i//1000}k' for i in range(
                int(df['salary'].min()), 
                int(df['salary'].max())+1, 
                50000
            )},
        )
    ], id='salary-slider-container', style={'display': 'none', 'padding': '20px'}),
    
    html.Div(id='error-message', style={'color': 'red', 'textAlign': 'center'}),
    
    html.Footer(
        f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        style={'textAlign': 'center', 'marginTop': '20px', 'padding': '10px'}
    )
])

# Callback to show/hide salary slider
@app.callback(
    Output('salary-slider-container', 'style'),
    Input('metric', 'value')
)
def toggle_salary_slider(metric):
    if metric == 'salary':
        return {'display': 'block', 'padding': '20px'}
    else:
        return {'display': 'none', 'padding': '20px'}

# Callback to update the graph based on selected metric
@app.callback(
    [Output('trend-graph', 'figure'),
     Output('error-message', 'children')],
    [Input('metric', 'value'),
     Input('country-selector', 'value'),
     Input('salary-range', 'value')]
)
def update_graph(metric, country, salary_range):
    try:
        # Filter data based on country selection
        if country != 'ALL':
            filtered_df = df[df['country'] == country].copy()
        else:
            filtered_df = df.copy()
        
        # Filter by salary range if applicable
        if metric == 'salary':
            filtered_df = filtered_df[
                (filtered_df['salary'] >= salary_range[0]) & 
                (filtered_df['salary'] <= salary_range[1])
            ]
        
        # Check if we have data after filtering
        if len(filtered_df) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for the selected filters",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig, "No data available for the selected filters"
        
        if metric == 'count':
            data = filtered_df.groupby('month_year').size().reset_index(name='value')
            title = f'Monthly Job Postings Trends - {country if country != "ALL" else "All Countries"}'
            y_axis_title = 'Number of Postings'
            fig = px.line(data, x='month_year', y='value', title=title)

        elif metric == 'salary':
            data = filtered_df.groupby('month_year')['salary'].mean().reset_index(name='value')
            title = f'Monthly Average Salary Trends - {country if country != "ALL" else "All Countries"}'
            y_axis_title = 'Average Salary ($)'
            fig = px.line(data, x='month_year', y='value', title=title)

        elif metric == 'remote_percentage':
            data = filtered_df.groupby('month_year')['is_remote'].mean().reset_index(name='value')
            data['value'] *= 100  # Convert to percentage
            title = f'Monthly Remote Work Percentage - {country if country != "ALL" else "All Countries"}'
            y_axis_title = 'Percentage (%)'
            fig = px.line(data, x='month_year', y='value', title=title)

        elif metric == 'salary_by_country':
            data = filtered_df.groupby(['month_year', 'country'])['salary'].mean().reset_index()
            title = 'Average Salary Trends by Country'
            y_axis_title = 'Average Salary ($)'
            fig = px.line(data, x='month_year', y='salary', color='country', title=title)

        # Common updates - FIXED: update_xaxes instead of update_xaxis
        fig.update_layout(
            xaxis_title='Month-Year',
            yaxis_title=y_axis_title,
            legend_title='Country' if metric == 'salary_by_country' else None,
            hovermode='x unified'
        )
        
        # Improve x-axis display - FIXED: update_xaxes instead of update_xaxis
        fig.update_xaxes(tickangle=45)
        
        return fig, ""  # No error message
        
    except Exception as e:
        error_msg = f"Error generating chart: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        
        # Create an empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text="Error loading data. Check console for details.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig, error_msg

if __name__ == '__main__':
    print("Starting Dash server...")
    app.run(debug=True, host='127.0.0.1', port=8050)