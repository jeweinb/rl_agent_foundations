"""
Shared layout components for the dashboard.
All dashboard layout files should use these instead of defining their own.
"""
from dash import html

from dashboard.theme import WHITE, GRAY_BORDER


CARD_STYLE = {
    "background": WHITE,
    "borderRadius": "12px",
    "padding": "20px",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
    "border": f"1px solid {GRAY_BORDER}",
    "flex": "1",
}


def card(children, **kwargs):
    """Reusable card container with consistent styling."""
    style = {**CARD_STYLE, **kwargs.pop("style", {})}
    return html.Div(children, style=style, **kwargs)


def row(children, gap="16px", **kwargs):
    """Flexbox row with consistent gap and margin."""
    style = {
        "display": "flex",
        "gap": gap,
        "marginBottom": "16px",
        **kwargs.pop("style", {}),
    }
    return html.Div(children, style=style, **kwargs)


def section_title(text, subtitle=None):
    """Section title with optional subtitle."""
    elements = [
        html.H3(text, style={
            "fontSize": "16px", "fontWeight": "600", "marginBottom": "4px",
        }),
    ]
    if subtitle:
        elements.append(html.P(subtitle, style={
            "fontSize": "12px", "color": "#64748b", "marginBottom": "8px",
        }))
    return html.Div(elements)
