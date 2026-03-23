"""Tab 6: Action Lifecycle State Machine — Sankey flow, funnel, transitions."""
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
        html.H2("Action Lifecycle (State Machine)", style={
            "fontSize": "20px", "fontWeight": "600", "marginBottom": "20px",
        }),

        # Sankey — hero
        _card([
            html.H3("Action Flow — Sankey Diagram", style={
                "fontSize": "16px", "fontWeight": "600", "marginBottom": "4px",
            }),
            html.P("Volume of actions flowing through each lifecycle state", style={
                "fontSize": "12px", "color": "#64748b", "marginBottom": "8px",
            }),
            dcc.Graph(id="sm-sankey", style={"height": "420px"},
                     config={"displayModeBar": False}),
        ], style={"flex": "none", "marginBottom": "16px"}),

        _row([
            _card([
                dcc.Graph(id="sm-funnel", style={"height": "340px"},
                         config={"displayModeBar": False}),
            ]),
            _card([
                dcc.Graph(id="sm-channel-funnel", style={"height": "340px"},
                         config={"displayModeBar": False}),
            ]),
        ]),

        _card([
            html.H3("Recent State Transitions", style={"fontSize": "16px", "fontWeight": "600", "marginBottom": "12px"}),
            html.Div(id="sm-transitions-table", style={
                "maxHeight": "300px", "overflowY": "auto",
            }),
        ], style={"flex": "none", "marginBottom": "16px"}),

        _card([
            dcc.Graph(id="sm-conversion-rates", style={"height": "300px"},
                     config={"displayModeBar": False}),
        ], style={"flex": "none"}),
    ])
