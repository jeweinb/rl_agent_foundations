"""Tab 2: Live System Behavior — actions, lifecycle, leaderboard, budget, Sankey."""
from dash import html, dcc
from dashboard.styles import card, row, section_title


def create_layout():
    return html.Div([
        html.H2("Live System Behavior", style={
            "fontSize": "20px", "fontWeight": "600", "marginBottom": "20px",
        }),

        # Global budget gauge
        card([
            section_title("Global Message Budget",
                         "Shared pool across all patients. The agent decides allocation."),
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

        # Sankey + distributions row
        row([
            card([
                section_title("Action Lifecycle Flow"),
                dcc.Graph(id="sm-sankey", style={"height": "380px"},
                         config={"displayModeBar": False}),
            ], style={"flex": "3"}),
            card([
                section_title("Action Funnel"),
                dcc.Graph(id="sm-funnel", style={"height": "380px"},
                         config={"displayModeBar": False}),
            ], style={"flex": "2"}),
        ]),

        # Channel / measure / action-vs-noaction
        row([
            card([dcc.Graph(id="action-by-channel", style={"height": "280px"},
                           config={"displayModeBar": False})]),
            card([dcc.Graph(id="action-by-measure", style={"height": "280px"},
                           config={"displayModeBar": False})]),
            card([dcc.Graph(id="action-vs-noaction", style={"height": "280px"},
                           config={"displayModeBar": False})]),
        ]),

        # Channel funnel + conversion rates
        row([
            card([dcc.Graph(id="sm-channel-funnel", style={"height": "300px"},
                           config={"displayModeBar": False})]),
            card([dcc.Graph(id="sm-conversion-rates", style={"height": "300px"},
                           config={"displayModeBar": False})]),
        ]),

        # Recent actions + transitions
        row([
            card([
                section_title("Recent Actions"),
                html.Div(id="recent-actions-table", style={"maxHeight": "300px", "overflowY": "auto"}),
            ]),
            card([
                section_title("Recent State Transitions"),
                html.Div(id="sm-transitions-table", style={"maxHeight": "300px", "overflowY": "auto"}),
            ]),
        ]),
    ])
