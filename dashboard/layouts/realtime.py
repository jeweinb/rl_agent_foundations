"""Tab 2: Real-Time Actions — live action feed, distribution charts, bubble."""
from dash import html, dcc
from dashboard.styles import card, row, section_title


def create_layout():
    return html.Div([
        html.H2("Real-Time Actions", style={
            "fontSize": "20px", "fontWeight": "600", "marginBottom": "20px",
        }),

        # Global budget / fatigue gauge
        card([
            section_title("Cohort Message Budget & Fatigue",
                         "Average budget remaining across all patients. Red = exhausted patients being suppressed."),
            html.Div(id="global-budget-gauge"),
        ], style={"flex": "none", "marginBottom": "16px"}),

        # Leaderboard with ranking toggle
        card([
            html.Div([
                section_title("Action Leaderboard",
                             "Which actions are most valuable? Toggle between model Q-values and observed outcomes."),
                html.Div([
                    html.Label("Rank by:", style={"fontWeight": "500", "marginRight": "8px"}),
                    dcc.Dropdown(
                        id="leaderboard-rank-by",
                        options=[
                            {"label": "Q-Value (Model's Predicted Future Reward)", "value": "q_value"},
                            {"label": "Completion Rate (State Machine)", "value": "completion"},
                            {"label": "Acceptance Rate (State Machine)", "value": "acceptance"},
                        ],
                        value="q_value",
                        style={"width": "400px", "display": "inline-block"},
                        clearable=False,
                    ),
                ], style={"marginBottom": "12px"}),
            ]),
            dcc.Graph(id="action-leaderboard", style={"height": "480px"},
                     config={"displayModeBar": False}),
        ], style={"flex": "none", "marginBottom": "16px"}),

        row([
            card([dcc.Graph(id="action-by-channel", style={"height": "280px"},
                           config={"displayModeBar": False})]),
            card([dcc.Graph(id="action-by-measure", style={"height": "280px"},
                           config={"displayModeBar": False})]),
            card([dcc.Graph(id="action-vs-noaction", style={"height": "280px"},
                           config={"displayModeBar": False})]),
        ]),

        card([
            section_title("Recent Actions"),
            html.Div(id="recent-actions-table", style={"maxHeight": "350px", "overflowY": "auto"}),
        ], style={"flex": "none"}),
    ])
