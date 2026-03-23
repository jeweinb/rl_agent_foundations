"""Tab 2: Real-Time Actions — live action feed, distribution charts, animated bubble."""
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
        html.H2("Real-Time Actions", style={
            "fontSize": "20px", "fontWeight": "600", "marginBottom": "20px",
        }),

        # Animated bubble chart — hero
        _card([
            html.H3("Patient Cohort Activity", style={
                "fontSize": "16px", "fontWeight": "600", "marginBottom": "4px",
            }),
            html.P("Bubble size = actions taken, color = primary measure targeted, position = engagement rate vs gaps remaining", style={
                "fontSize": "12px", "color": "#64748b", "marginBottom": "8px",
            }),
            dcc.Graph(id="cohort-bubble", style={"height": "380px"},
                     config={"displayModeBar": False}),
        ], style={"flex": "none", "marginBottom": "16px"}),

        # Charts row
        _row([
            _card([
                dcc.Graph(id="action-by-channel", style={"height": "280px"},
                         config={"displayModeBar": False}),
            ]),
            _card([
                dcc.Graph(id="action-by-measure", style={"height": "280px"},
                         config={"displayModeBar": False}),
            ]),
            _card([
                dcc.Graph(id="action-vs-noaction", style={"height": "280px"},
                         config={"displayModeBar": False}),
            ]),
        ]),

        # Recent actions table
        _card([
            html.H3("Recent Actions", style={"fontSize": "16px", "fontWeight": "600", "marginBottom": "12px"}),
            html.Div(id="recent-actions-table", style={
                "maxHeight": "350px", "overflowY": "auto",
            }),
        ], style={"flex": "none"}),
    ])
