"""Tab 3: Training Performance — champion vs challenger, model versions."""
from dash import html, dcc
from dashboard.styles import card, row, section_title


def create_layout():
    return html.Div([
        html.H2("Training Performance", style={
            "fontSize": "20px", "fontWeight": "600", "marginBottom": "20px",
        }),
        row([
            card([dcc.Graph(id="champion-challenger", style={"height": "380px"},
                           config={"displayModeBar": False})]),
            card([dcc.Graph(id="model-version-timeline", style={"height": "380px"},
                           config={"displayModeBar": False})]),
        ]),
        card([
            section_title("Promotion History"),
            html.Div(id="promotion-history-table"),
        ], style={"flex": "none"}),
    ])
