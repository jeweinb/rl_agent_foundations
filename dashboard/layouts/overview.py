"""Tab 1: STARS Overview — gauge, trajectory, measure table, cumulative reward."""
from dash import html, dcc
from dashboard.styles import card, row, section_title


def create_layout():
    return html.Div([
        html.H2("STARS Performance Overview", style={
            "fontSize": "20px", "fontWeight": "600", "marginBottom": "20px",
        }),
        row([
            card([dcc.Graph(id="stars-gauge", style={"height": "280px"},
                           config={"displayModeBar": False})], style={"flex": "2"}),
            card([dcc.Graph(id="stars-trajectory", style={"height": "280px"},
                           config={"displayModeBar": False})], style={"flex": "3"}),
        ]),
        row([
            card([dcc.Graph(id="cumulative-reward", style={"height": "260px"},
                           config={"displayModeBar": False})]),
            card([dcc.Graph(id="regret-curve", style={"height": "260px"},
                           config={"displayModeBar": False})]),
        ]),
        card([
            section_title("Gap Closure Heatmap — Daily Intensity by Measure",
                         "Color intensity shows gap closure rate progression per measure per day"),
            dcc.Graph(id="closure-heatmap", style={"height": "380px"},
                     config={"displayModeBar": False}),
        ], style={"flex": "none", "marginBottom": "16px"}),
        card([
            section_title("Measure Gap Closure Rates"),
            html.Div(id="measure-table"),
        ], style={"flex": "none"}),
    ])
