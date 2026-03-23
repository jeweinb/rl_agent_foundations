"""Tab 3: Training Performance — champion vs challenger, model versions."""
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
        html.H2("Training Performance", style={
            "fontSize": "20px", "fontWeight": "600", "marginBottom": "20px",
        }),
        _row([
            _card([
                dcc.Graph(id="champion-challenger", style={"height": "380px"},
                         config={"displayModeBar": False}),
            ]),
            _card([
                dcc.Graph(id="model-version-timeline", style={"height": "380px"},
                         config={"displayModeBar": False}),
            ]),
        ]),
        _card([
            html.H3("Promotion History", style={"fontSize": "16px", "fontWeight": "600", "marginBottom": "12px"}),
            html.Div(id="promotion-history-table"),
        ], style={"flex": "none"}),
    ])
