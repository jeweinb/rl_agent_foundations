"""
Plotly Dash application entry point.
7-tab dashboard with real-time streaming updates and modern UI.
"""
from dash import Dash, html, dcc

from config import DASHBOARD_UPDATE_INTERVAL_MS
from dashboard.layouts import overview, realtime, training, measures, patient_journey, logs
from dashboard.callbacks import register_callbacks


# Modern CSS styles
GLOBAL_STYLES = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background-color: #F5F6F8;
    margin: 0;
    color: #1B2A4A;
}

.tab-content {
    padding: 24px 32px;
    width: 100%;
    box-sizing: border-box;
    overflow-x: hidden;
}

/* Fix flex items not shrinking on zoom-out */
.tab-content > div {
    min-width: 0;
}

/* Ensure flex rows wrap at small widths */
.tab-content div[style*="display: flex"] {
    flex-wrap: wrap !important;
    min-width: 0;
}

/* Cards should have min-width to prevent collapse but allow shrink */
.tab-content div[style*="border-radius: 12px"] {
    min-width: 200px;
    box-sizing: border-box;
}

table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    font-size: 13px;
}

table thead tr {
    background: linear-gradient(135deg, #1B2A4A, #2D4263);
    color: white;
}

table thead th {
    padding: 12px 16px;
    text-align: left;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

table tbody tr {
    border-bottom: 1px solid #edf2f7;
    transition: background-color 0.15s;
}

table tbody tr:hover {
    background-color: #f7fafc !important;
}

table tbody td {
    padding: 10px 16px;
    vertical-align: middle;
}

table tbody tr:nth-child(even) {
    background-color: #fafbfc;
}

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border: 1px solid #e2e8f0;
}

.section-title {
    font-size: 18px;
    font-weight: 600;
    color: #1a1a2e;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid #e2e8f0;
}

.stat-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}
"""


def create_app() -> Dash:
    """Create and configure the Dash application."""
    app = Dash(
        __name__,
        title="NBA Stars Model — CQL Offline RL Agent",
        suppress_callback_exceptions=True,
    )

    app.index_string = '''<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>''' + GLOBAL_STYLES + '''</style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>'''

    tab_style = {
        "padding": "12px 24px",
        "fontWeight": "500",
        "fontSize": "13px",
        "borderBottom": "2px solid transparent",
        "color": "#64748b",
    }
    tab_selected_style = {
        **tab_style,
        "color": "#1B2A4A",
        "borderBottom": "3px solid #00A664",
        "fontWeight": "600",
    }

    app.layout = html.Div([
        # Header
        html.Div([
            html.Div([
                html.H1("NBA Stars Model", style={
                    "margin": "0", "fontSize": "22px", "fontWeight": "700",
                    "color": "#1B2A4A", "display": "inline-block",
                }),
                html.P("CQL Offline RL Agent | Monitoring Dashboard", style={
                    "margin": "2px 0 0", "fontSize": "13px", "color": "#64748b",
                }),
            ], style={"display": "inline-block"}),
            # Day counter (top right)
            html.Div(id="day-counter", style={
                "float": "right", "textAlign": "right", "paddingTop": "4px",
            }),
        ], style={
            "backgroundColor": "white", "padding": "16px 32px",
            "borderBottom": "1px solid #e2e8f0",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
        }),

        # Interval for real-time updates
        dcc.Interval(
            id="interval-component",
            interval=DASHBOARD_UPDATE_INTERVAL_MS,
            n_intervals=0,
        ),

        # Tabs
        dcc.Tabs(id="tabs", value="tab-overview", children=[
            dcc.Tab(label="STARS Overview", value="tab-overview",
                   children=html.Div(overview.create_layout(), className="tab-content"),
                   style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label="Live Behavior", value="tab-realtime",
                   children=html.Div(realtime.create_layout(), className="tab-content"),
                   style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label="Training & Simulation", value="tab-training",
                   children=html.Div(training.create_layout(), className="tab-content"),
                   style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label="Measures", value="tab-measures",
                   children=html.Div(measures.create_layout(), className="tab-content"),
                   style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label="Patient Journey", value="tab-patient",
                   children=html.Div(patient_journey.create_layout(), className="tab-content"),
                   style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label="Logs", value="tab-logs",
                   children=html.Div(logs.create_layout(), className="tab-content"),
                   style=tab_style, selected_style=tab_selected_style),
        ], style={
            "backgroundColor": "white", "borderBottom": "1px solid #e2e8f0",
            "padding": "0 24px",
        }),
    ], style={"minHeight": "100vh", "backgroundColor": "#f0f2f5"})

    register_callbacks(app)
    return app
