"""
Centralized color palette and theming constants for the dashboard.
All dashboard modules should import colors from here — never hardcode hex values.
"""

# --- Brand Colors ---
HUMANA_GREEN = "#00A664"
HUMANA_DARK_GREEN = "#007A4D"
NAVY = "#1B2A4A"
NAVY_LIGHT = "#2D4263"

# --- UI Colors ---
WHITE = "#FFFFFF"
GRAY_BG = "#F5F6F8"
GRAY_BORDER = "#E2E8F0"
GRAY_TEXT = "#64748b"
GRAY_LIGHT = "#f1f5f9"

# --- Status Colors ---
STATUS_SUCCESS = "#22c55e"
STATUS_SUCCESS_BG = "#d1fae5"
STATUS_WARNING = "#f59e0b"
STATUS_WARNING_BG = "#fef3c7"
STATUS_ERROR = "#ef4444"
STATUS_ERROR_BG = "#fee2e2"
STATUS_CRITICAL = "#ea580c"
STATUS_CRITICAL_BG = "#ffedd5"

# --- Channel Colors ---
CHANNEL_COLORS = {
    "sms": "#1f77b4",
    "email": "#ff7f0e",
    "portal": "#2ca02c",
    "app": "#d62728",
    "ivr": "#9467bd",
    "none": "#999999",
}

CHANNEL_BG_COLORS = {
    "sms": "#e3f2fd",
    "email": "#fff3e0",
    "portal": "#e8f5e9",
    "app": "#fce4ec",
    "ivr": "#f3e5f5",
}

CHANNEL_ICONS = {
    "sms": "📱",
    "email": "📧",
    "portal": "🌐",
    "app": "📲",
    "ivr": "📞",
    None: "⏸️",
}

# --- Chart Constants ---
CHART_TEMPLATE = "plotly_white"
CHART_FONT = dict(family="Inter, sans-serif", size=12, color=NAVY)
CHART_MARGIN = dict(l=48, r=16, t=48, b=40)

# --- Sankey / Funnel State Colors ---
STATE_COLORS = {
    "CREATED": "#94a3b8",
    "QUEUED": "#64748b",
    "PRESENTED": "#3b82f6",
    "VIEWED": "#8b5cf6",
    "ACCEPTED": HUMANA_GREEN,
    "COMPLETED": "#15803d",
    "DECLINED": "#f59e0b",
    "FAILED": "#ef4444",
    "EXPIRED": "#9ca3af",
}
