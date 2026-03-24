"""Tab 4: Measure Deep Dive — per-measure gap closure, channel effectiveness, chord."""
from dash import html, dcc
from config import HEDIS_MEASURES, MEASURE_DESCRIPTIONS
from dashboard.styles import card, row, section_title


def create_layout():
    options = [{"label": f"{m} — {MEASURE_DESCRIPTIONS.get(m, m)}", "value": m} for m in HEDIS_MEASURES]

    return html.Div([
        html.H2("Measure Deep Dive", style={
            "fontSize": "20px", "fontWeight": "600", "marginBottom": "20px",
        }),
        card([
            section_title("Channel × Measure Effectiveness",
                         "Heatmap showing click-through rate for each channel-measure combination"),
            dcc.Graph(id="channel-measure-chord", style={"height": "420px"},
                     config={"displayModeBar": False}),
        ], style={"flex": "none", "marginBottom": "16px"}),
        card([
            html.Label("Select HEDIS Measure:", style={"fontWeight": "500", "marginBottom": "8px", "display": "block"}),
            dcc.Dropdown(id="measure-selector", options=options, value="COL", style={"width": "450px"}),
        ], style={"flex": "none", "marginBottom": "16px"}),
        row([
            card([dcc.Graph(id="measure-closure-trend", style={"height": "340px"},
                           config={"displayModeBar": False})]),
            card([dcc.Graph(id="measure-channel-effectiveness", style={"height": "340px"},
                           config={"displayModeBar": False})]),
        ]),
        card([dcc.Graph(id="measure-funnel", style={"height": "320px"},
                       config={"displayModeBar": False})], style={"flex": "none"}),
    ])
