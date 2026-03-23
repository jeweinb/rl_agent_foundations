"""Tab 4: Measure Deep Dive — per-measure gap closure, channel effectiveness, chord diagram."""
from dash import html, dcc
from config import HEDIS_MEASURES, MEASURE_DESCRIPTIONS


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
    options = [{"label": f"{m} — {MEASURE_DESCRIPTIONS.get(m, m)}", "value": m} for m in HEDIS_MEASURES]

    return html.Div([
        html.H2("Measure Deep Dive", style={
            "fontSize": "20px", "fontWeight": "600", "marginBottom": "20px",
        }),

        # Channel × Measure Chord / Heatmap — hero
        _card([
            html.H3("Channel × Measure Effectiveness", style={
                "fontSize": "16px", "fontWeight": "600", "marginBottom": "4px",
            }),
            html.P("Heatmap showing click-through rate for each channel-measure combination", style={
                "fontSize": "12px", "color": "#64748b", "marginBottom": "8px",
            }),
            dcc.Graph(id="channel-measure-chord", style={"height": "420px"},
                     config={"displayModeBar": False}),
        ], style={"flex": "none", "marginBottom": "16px"}),

        _card([
            html.Label("Select HEDIS Measure:", style={"fontWeight": "500", "marginBottom": "8px", "display": "block"}),
            dcc.Dropdown(
                id="measure-selector",
                options=options,
                value="COL",
                style={"width": "450px"},
            ),
        ], style={"flex": "none", "marginBottom": "16px"}),

        _row([
            _card([
                dcc.Graph(id="measure-closure-trend", style={"height": "340px"},
                         config={"displayModeBar": False}),
            ]),
            _card([
                dcc.Graph(id="measure-channel-effectiveness", style={"height": "340px"},
                         config={"displayModeBar": False}),
            ]),
        ]),

        _card([
            dcc.Graph(id="measure-funnel", style={"height": "320px"},
                     config={"displayModeBar": False}),
        ], style={"flex": "none"}),
    ])
