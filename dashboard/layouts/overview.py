"""Tab 1: STARS Overview — gauge, trajectory, measure table, cumulative reward."""
from dash import html, dcc


def _card(children, **kwargs):
    style = {
        "background": "white", "borderRadius": "12px", "padding": "20px",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.08)", "border": "1px solid #e2e8f0",
        "flex": "1",
        **kwargs.pop("style", {}),
    }
    return html.Div(children, style=style, **kwargs)


def _row(children, gap="16px", **kwargs):
    return html.Div(children, style={
        "display": "flex", "gap": gap, "marginBottom": "16px",
        **kwargs.pop("style", {}),
    }, **kwargs)


def create_layout():
    return html.Div([
        html.H2("STARS Performance Overview", style={
            "fontSize": "20px", "fontWeight": "600", "marginBottom": "20px",
        }),

        # Top row: Gauge + Trajectory
        _row([
            _card([
                dcc.Graph(id="stars-gauge", style={"height": "280px"},
                         config={"displayModeBar": False}),
            ], style={"flex": "2"}),
            _card([
                dcc.Graph(id="stars-trajectory", style={"height": "280px"},
                         config={"displayModeBar": False}),
            ], style={"flex": "3"}),
        ]),

        # Middle row: Cumulative Reward + Regret
        _row([
            _card([
                dcc.Graph(id="cumulative-reward", style={"height": "260px"},
                         config={"displayModeBar": False}),
            ]),
            _card([
                dcc.Graph(id="regret-curve", style={"height": "260px"},
                         config={"displayModeBar": False}),
            ]),
        ]),

        # Heatmap calendar
        _card([
            html.H3("Gap Closure Heatmap — Daily Intensity by Measure", style={
                "fontSize": "16px", "fontWeight": "600", "marginBottom": "4px",
            }),
            html.P("Color intensity shows gap closure rate progression per measure per day", style={
                "fontSize": "12px", "color": "#64748b", "marginBottom": "8px",
            }),
            dcc.Graph(id="closure-heatmap", style={"height": "380px"},
                     config={"displayModeBar": False}),
        ], style={"marginBottom": "16px", "flex": "none"}),

        # Measure table
        _card([
            html.H3("Measure Gap Closure Rates", style={
                "fontSize": "16px", "fontWeight": "600", "marginBottom": "12px",
            }),
            html.Div(id="measure-table"),
        ], style={"flex": "none"}),
    ])
