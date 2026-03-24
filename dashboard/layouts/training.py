"""Tab 3: Training & Simulation — CQL training performance + simulated next-day predictions."""
from dash import html, dcc
from dashboard.styles import card, row, section_title


def create_layout():
    return html.Div([
        html.H2("Training & Simulation", style={
            "fontSize": "20px", "fontWeight": "600", "marginBottom": "20px",
        }),

        # Top row: Champion vs Challenger + Model Version Timeline
        row([
            card([dcc.Graph(id="champion-challenger", style={"height": "340px"},
                           config={"displayModeBar": False})]),
            card([dcc.Graph(id="model-version-timeline", style={"height": "340px"},
                           config={"displayModeBar": False})]),
        ]),

        # Simulated next-day predictions from learned world
        card([
            section_title("Simulated Performance (Learned World Predictions)",
                         "What the dynamics + reward models predict will happen if we deploy the current champion tomorrow"),
            html.Div(id="sim-performance-summary", style={"marginBottom": "16px"}),
        ], style={"flex": "none", "marginBottom": "16px"}),

        # Simulated action distribution + closure predictions
        row([
            card([
                section_title("Predicted Action Distribution"),
                dcc.Graph(id="sim-action-distribution", style={"height": "320px"},
                         config={"displayModeBar": False}),
            ]),
            card([
                section_title("Predicted Gap Closures by Measure"),
                dcc.Graph(id="sim-closure-predictions", style={"height": "320px"},
                         config={"displayModeBar": False}),
            ]),
        ]),

        row([
            card([
                section_title("Predicted Channel Effectiveness"),
                dcc.Graph(id="sim-channel-effectiveness", style={"height": "300px"},
                         config={"displayModeBar": False}),
            ]),
            card([
                section_title("Simulated STARS Projection"),
                dcc.Graph(id="sim-stars-projection", style={"height": "300px"},
                         config={"displayModeBar": False}),
            ]),
        ]),

        # Promotion history
        card([
            section_title("Model Promotion History"),
            html.Div(id="promotion-history-table"),
        ], style={"flex": "none"}),
    ])
