"""Tab 7: Simulation Logs — real-time log stream from the simulation."""
from dash import html, dcc


def create_layout():
    return html.Div([
        html.H2("Simulation Logs", style={
            "fontSize": "20px", "fontWeight": "600", "marginBottom": "20px",
        }),
        html.Div([
            html.Label("Filter:", style={"fontWeight": "500", "marginRight": "12px"}),
            dcc.Dropdown(
                id="log-level-filter",
                options=[
                    {"label": "All", "value": "ALL"},
                    {"label": "Phases", "value": "PHASE"},
                    {"label": "Metrics", "value": "METRIC"},
                    {"label": "Info", "value": "INFO"},
                    {"label": "Errors", "value": "ERROR"},
                ],
                value="ALL",
                style={"width": "180px", "display": "inline-block"},
            ),
        ], style={"marginBottom": "16px"}),
        html.Div([
            html.Div(
                id="log-stream",
                style={
                    "backgroundColor": "#0d1117",
                    "color": "#c9d1d9",
                    "fontFamily": "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
                    "fontSize": "12px",
                    "padding": "20px",
                    "borderRadius": "12px",
                    "maxHeight": "600px",
                    "overflowY": "auto",
                    "whiteSpace": "pre-wrap",
                    "lineHeight": "1.6",
                    "border": "1px solid #21262d",
                },
            ),
        ], style={
            "background": "white", "borderRadius": "12px", "padding": "4px",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.08)", "border": "1px solid #e2e8f0",
        }),
    ])
