"""Tab 5: Patient Journey — interactive timeline grouped by week with action cards."""
from dash import html, dcc
from dashboard.styles import card, row, section_title


def create_layout():
    return html.Div([
        html.H2("Patient Journey", style={
            "fontSize": "20px", "fontWeight": "600", "marginBottom": "20px",
        }),

        # Patient selector + summary
        card([
            html.Div([
                html.Label("Search Patient ID:", style={"fontWeight": "500", "marginBottom": "8px", "display": "block"}),
                dcc.Dropdown(
                    id="patient-selector",
                    placeholder="Select or search patient...",
                    style={"width": "300px"},
                ),
            ], style={"marginBottom": "16px"}),
            html.Div(id="patient-summary"),
        ], style={"flex": "none", "marginBottom": "16px"}),

        # Budget / contact intensity
        html.Div(id="patient-budget-bar", style={"marginBottom": "16px"}),

        # Weekly action timeline
        card([
            section_title("Action Timeline", "Actions grouped by week. Only weeks with activity shown."),
            html.Div(id="patient-action-cards", style={
                "maxHeight": "500px", "overflowY": "auto", "padding": "4px",
            }),
        ], style={"flex": "none", "marginBottom": "16px"}),

        # Charts row
        row([
            card([dcc.Graph(id="patient-reward-curve", style={"height": "260px"},
                           config={"displayModeBar": False})]),
            card([dcc.Graph(id="patient-gap-status", style={"height": "260px"},
                           config={"displayModeBar": False})]),
        ]),
    ])
