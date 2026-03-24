"""Tab 4: Measure Deep Dive — full measure-specific detail.

When a measure is selected, shows:
- Measure overview card (description, weight, current rate, star rating, 4★ target)
- Actions deployed for this measure: which action variants, how many sent, acceptance rate
- Patient engagement: how many patients have this gap, how many were contacted
- Action effectiveness table: rank all actions for this measure by closure contribution
- Closure trend over time with CMS cut points
- Channel × variant heatmap for this specific measure
"""
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
            html.Label("Select HEDIS Measure:", style={"fontWeight": "500", "marginBottom": "8px", "display": "block"}),
            dcc.Dropdown(id="measure-selector", options=options, value="COL", style={"width": "450px"}),
        ], style={"flex": "none", "marginBottom": "16px"}),

        # Measure overview card
        html.Div(id="measure-overview-card", style={"marginBottom": "16px"}),

        # Closure trend + patients reached
        row([
            card([
                section_title("Gap Closure Trend"),
                dcc.Graph(id="measure-closure-trend", style={"height": "320px"},
                         config={"displayModeBar": False}),
            ]),
            card([
                section_title("Action Variant Performance",
                             "Which specific actions are working best for this measure?"),
                dcc.Graph(id="measure-action-variants", style={"height": "320px"},
                         config={"displayModeBar": False}),
            ]),
        ]),

        # Channel effectiveness for this measure + action table
        row([
            card([
                section_title("Channel Effectiveness"),
                dcc.Graph(id="measure-channel-effectiveness", style={"height": "300px"},
                         config={"displayModeBar": False}),
            ]),
            card([
                section_title("Action Deployment Summary",
                             "All actions sent for this measure with volume and outcomes"),
                html.Div(id="measure-action-table", style={"maxHeight": "300px", "overflowY": "auto"}),
            ]),
        ]),
    ])
