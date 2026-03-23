"""Tab 5: Patient Journey — rich interactive timeline with action cards."""
from dash import html, dcc


CHANNEL_ICONS = {
    "sms": "📱", "email": "📧", "portal": "🌐", "app": "📲", "ivr": "📞", None: "⏸️",
}

CHANNEL_COLORS = {
    "sms": "#1f77b4", "email": "#ff7f0e", "portal": "#2ca02c",
    "app": "#d62728", "ivr": "#9467bd", None: "#999999",
}


def create_layout():
    return html.Div([
        html.H2("Patient Journey"),
        html.Div([
            html.Label("Search Patient ID:"),
            dcc.Dropdown(
                id="patient-selector",
                placeholder="Select or search patient...",
                style={"width": "300px"},
            ),
        ], style={"marginBottom": "20px"}),
        html.Div(id="patient-summary", style={"marginBottom": "20px"}),

        # Budget bar
        html.Div(id="patient-budget-bar", style={"marginBottom": "20px"}),

        # Interactive action cards timeline
        html.Div([
            html.H3("Action Timeline"),
            html.Div(id="patient-action-cards", style={
                "display": "flex", "flexWrap": "wrap", "gap": "12px",
                "maxHeight": "500px", "overflowY": "auto", "padding": "10px",
            }),
        ]),

        # Charts row
        html.Div([
            html.Div([
                dcc.Graph(id="patient-reward-curve", style={"height": "250px"}),
            ], style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
            html.Div([
                dcc.Graph(id="patient-gap-status", style={"height": "250px"}),
            ], style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
        ]),
    ])
