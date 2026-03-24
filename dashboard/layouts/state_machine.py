"""Tab 6: Action Lifecycle State Machine — Sankey flow, funnel, transitions."""
from dash import html, dcc
from dashboard.styles import card, row, section_title


def create_layout():
    return html.Div([
        html.H2("Action Lifecycle (State Machine)", style={
            "fontSize": "20px", "fontWeight": "600", "marginBottom": "20px",
        }),
        card([
            section_title("Action Flow — Sankey Diagram",
                         "Volume of actions flowing through each lifecycle state"),
            dcc.Graph(id="sm-sankey", style={"height": "420px"},
                     config={"displayModeBar": False}),
        ], style={"flex": "none", "marginBottom": "16px"}),
        row([
            card([dcc.Graph(id="sm-funnel", style={"height": "340px"},
                           config={"displayModeBar": False})]),
            card([dcc.Graph(id="sm-channel-funnel", style={"height": "340px"},
                           config={"displayModeBar": False})]),
        ]),
        card([
            section_title("Recent State Transitions"),
            html.Div(id="sm-transitions-table", style={"maxHeight": "300px", "overflowY": "auto"}),
        ], style={"flex": "none", "marginBottom": "16px"}),
        card([dcc.Graph(id="sm-conversion-rates", style={"height": "300px"},
                       config={"displayModeBar": False})], style={"flex": "none"}),
    ])
