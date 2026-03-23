"""
Dash callbacks for all dashboard tabs.
Handles real-time data updates via dcc.Interval polling.
"""
from dash import Input, Output, State, callback, html, dash_table
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter

from config import HEDIS_MEASURES, MEASURE_DESCRIPTIONS, MEASURE_WEIGHTS, STARS_BONUS_THRESHOLD

# --- Humana Brand Colors ---
HUMANA_GREEN = "#00A664"
HUMANA_DARK_GREEN = "#007A4D"
NAVY = "#1B2A4A"
NAVY_LIGHT = "#2D4263"
GRAY_BG = "#F5F6F8"
GRAY_BORDER = "#E2E8F0"
WHITE = "#FFFFFF"

# Chart template
CHART_TEMPLATE = "plotly_white"
CHART_FONT = dict(family="Inter, sans-serif", size=12, color=NAVY)
CHART_MARGIN = dict(l=48, r=16, t=48, b=40)

def _styled_fig(fig):
    """Apply consistent styling to all charts."""
    fig.update_layout(
        template=CHART_TEMPLATE,
        font=CHART_FONT,
        margin=CHART_MARGIN,
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
    )
    return fig


def _empty_fig(title="Waiting for simulation data..."):
    """Create a styled empty placeholder figure."""
    fig = go.Figure()
    fig.add_annotation(
        text=title, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="#94a3b8"),
    )
    fig.update_layout(
        template=CHART_TEMPLATE,
        font=CHART_FONT,
        margin=CHART_MARGIN,
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig
from dashboard.data_feed import (
    load_cumulative_metrics, load_all_actions, load_all_nightly_metrics,
    get_patient_journey, get_all_patient_ids, get_latest_day,
    load_all_state_machine_data, load_simulation_logs,
)


def register_callbacks(app):
    """Register all callbacks with the Dash app."""

    # =========================================================================
    # Tab 1: STARS Overview
    # =========================================================================
    @app.callback(
        [Output("stars-gauge", "figure"),
         Output("stars-trajectory", "figure"),
         Output("cumulative-reward", "figure"),
         Output("regret-curve", "figure"),
         Output("closure-heatmap", "figure"),
         Output("measure-table", "children")],
        Input("interval-component", "n_intervals"),
    )
    def update_overview(_):
        metrics = load_cumulative_metrics()

        # STARS Gauge
        stars_score = metrics[-1]["stars_score"] if metrics else 1.0
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=stars_score,
            title={"text": "Projected STARS Score", "font": {"size": 16, "color": NAVY}},
            delta={"reference": STARS_BONUS_THRESHOLD, "increasing": {"color": HUMANA_GREEN},
                   "prefix": "", "suffix": " to bonus", "valueformat": "+.1f"},
            gauge={
                "axis": {"range": [1, 5], "tickwidth": 2, "tickcolor": NAVY},
                "bar": {"color": HUMANA_GREEN},
                "steps": [
                    {"range": [1, 2], "color": "#fee2e2"},
                    {"range": [2, 3], "color": "#fef3c7"},
                    {"range": [3, 4], "color": "#d1fae5"},
                    {"range": [4, 5], "color": "#a7f3d0"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 4},
                    "thickness": 0.75,
                    "value": STARS_BONUS_THRESHOLD,
                },
            },
        ))
        _styled_fig(gauge)
        gauge.update_layout(margin=dict(l=20, r=20, t=50, b=20))

        # STARS Trajectory
        if metrics:
            days = [m["day"] for m in metrics]
            scores = [m["stars_score"] for m in metrics]
            traj = go.Figure()
            traj.add_trace(go.Scatter(x=days, y=scores, mode="lines+markers", name="STARS Score",
                                      line=dict(color=HUMANA_GREEN, width=3),
                                      marker=dict(size=8, color=HUMANA_GREEN)))
            traj.add_hline(y=STARS_BONUS_THRESHOLD, line_dash="dash", line_color="#ef4444",
                          annotation_text="4.0 Bonus Threshold")
            _styled_fig(traj)
            traj.update_layout(title="STARS Score Over Time", xaxis_title="Day", yaxis_title="Score",
                             yaxis=dict(range=[1, 5]))
        else:
            traj = _empty_fig("STARS Score Over Time — waiting for data...")

        # Cumulative Reward
        if metrics:
            cum_rewards = [m["cumulative_reward"] for m in metrics]
            cum = go.Figure()
            cum.add_trace(go.Scatter(x=days, y=cum_rewards, mode="lines+markers",
                                     fill="tozeroy", name="Cumulative Reward",
                                     line=dict(color=HUMANA_GREEN, width=2),
                                     fillcolor="rgba(0,166,100,0.1)"))
            _styled_fig(cum)
            cum.update_layout(title="Cumulative Reward", xaxis_title="Day", yaxis_title="Reward")
        else:
            cum = _empty_fig("Cumulative Reward — waiting for data...")

        # Regret Curve
        if metrics:
            oracle_per_day = max(m["daily_reward"] for m in metrics) * 1.5 if metrics else 5.0
            regret = []
            cum_regret = 0.0
            for m in metrics:
                cum_regret += oracle_per_day - m["daily_reward"]
                regret.append(cum_regret)
            reg = go.Figure()
            reg.add_trace(go.Scatter(x=days, y=regret, mode="lines", name="Cumulative Regret",
                                     line=dict(color=NAVY, width=2)))
            _styled_fig(reg)
            reg.update_layout(title="Cumulative Regret vs Oracle", xaxis_title="Day",
                            yaxis_title="Regret")
        else:
            reg = _empty_fig("Regret Curve — waiting for data...")

        # Measure Table — uses CMS methodology with per-measure star ratings
        if metrics:
            latest = metrics[-1]
            measure_detail = latest.get("measure_detail", {})
            closure_rates = latest.get("measure_closure_rates", {})
            rows = []
            for m in HEDIS_MEASURES:
                detail = measure_detail.get(m, {})
                rate = detail.get("rate", closure_rates.get(m, 0))
                stars = detail.get("stars", 1.0)
                threshold = detail.get("threshold_4star", 0.70)
                gap_to_4 = detail.get("gap_to_4star", threshold - rate)
                weight = detail.get("weight", MEASURE_WEIGHTS.get(m, 1))
                at4 = detail.get("at_or_above_4", False)

                # Star rating badge color
                if stars >= 4.0:
                    star_color, star_bg = HUMANA_GREEN, "#d1fae5"
                elif stars >= 3.0:
                    star_color, star_bg = "#d97706", "#fef3c7"
                elif stars >= 2.0:
                    star_color, star_bg = "#ea580c", "#ffedd5"
                else:
                    star_color, star_bg = "#dc2626", "#fee2e2"

                # Rate vs threshold progress bar
                pct = min(rate / max(threshold, 0.01) * 100, 100)
                bar_color = HUMANA_GREEN if rate >= threshold else ("#f59e0b" if pct > 60 else "#ef4444")

                progress_bar = html.Div([
                    html.Div(style={
                        "width": f"{pct:.0f}%", "height": "8px",
                        "backgroundColor": bar_color, "borderRadius": "4px",
                    }),
                ], style={
                    "width": "100%", "backgroundColor": "#e2e8f0",
                    "borderRadius": "4px", "overflow": "hidden", "minWidth": "100px",
                })

                weight_badge = html.Span(
                    f"{weight}x",
                    style={"backgroundColor": NAVY if weight > 1 else "#e2e8f0",
                           "color": "white" if weight > 1 else "#64748b",
                           "padding": "2px 10px", "borderRadius": "12px",
                           "fontSize": "11px", "fontWeight": "600"},
                )
                star_badge = html.Span(
                    f"{'★' * int(stars)}{'☆' * (5 - int(stars))} {stars:.1f}",
                    style={"color": star_color, "fontWeight": "600", "fontSize": "12px"},
                )
                rows.append(html.Tr([
                    html.Td(html.Span(m, style={"fontWeight": "600"})),
                    html.Td(MEASURE_DESCRIPTIONS.get(m, ""), style={"color": "#64748b", "fontSize": "12px"}),
                    html.Td(weight_badge),
                    html.Td(f"{rate:.1%}", style={"fontWeight": "500"}),
                    html.Td(f"{threshold:.0%}", style={"color": "#64748b", "fontSize": "12px"}),
                    html.Td(progress_bar),
                    html.Td(star_badge),
                ]))
            table = html.Table([
                html.Thead(html.Tr([
                    html.Th("Measure"), html.Th("Description"), html.Th("Wt"),
                    html.Th("Rate"), html.Th("4★ Target"), html.Th("Progress"), html.Th("Stars"),
                ])),
                html.Tbody(rows),
            ], style={"width": "100%", "borderCollapse": "collapse"})
        else:
            table = html.P("Waiting for simulation data...")

        # --- Heatmap: Daily gap closure intensity by measure ---
        if metrics and len(metrics) > 1:
            days_list = [m["day"] for m in metrics]
            measures_with_data = [m for m in HEDIS_MEASURES if any(
                met.get("measure_closure_rates", {}).get(m, 0) > 0 for met in metrics
            )]
            if not measures_with_data:
                measures_with_data = HEDIS_MEASURES[:8]
            z_data = []
            for m in measures_with_data:
                row = [met.get("measure_closure_rates", {}).get(m, 0) for met in metrics]
                z_data.append(row)
            heatmap = go.Figure(go.Heatmap(
                z=z_data, x=days_list, y=measures_with_data,
                colorscale=[[0, "#f0fdf4"], [0.3, "#86efac"], [0.6, "#22c55e"], [1.0, "#15803d"]],
                hoverongaps=False,
                colorbar=dict(title="Rate", thickness=15),
            ))
            _styled_fig(heatmap)
            heatmap.update_layout(
                title="Gap Closure Rate by Measure Over Time",
                xaxis_title="Simulation Day", yaxis_title="",
                yaxis=dict(dtick=1),
            )
        else:
            heatmap = _empty_fig("Gap Closure Heatmap — waiting for data...")

        return gauge, traj, cum, reg, heatmap, table

    # =========================================================================
    # Tab 2: Real-Time Actions
    # =========================================================================
    @app.callback(
        [Output("cohort-bubble", "figure"),
         Output("recent-actions-table", "children"),
         Output("action-by-channel", "figure"),
         Output("action-by-measure", "figure"),
         Output("action-vs-noaction", "figure")],
        Input("interval-component", "n_intervals"),
    )
    def update_realtime(_):
        actions = load_all_actions()

        # Recent actions table (last 50)
        recent = actions[-50:] if actions else []
        if recent:
            rows = []
            for a in reversed(recent):
                eng = a.get("engagement", {})
                status = "Clicked" if eng.get("clicked") else ("Viewed" if eng.get("opened") else ("Delivered" if eng.get("delivered") else "—"))
                rows.append(html.Tr([
                    html.Td(f"Day {a.get('day', '?')}"),
                    html.Td(a.get("patient_id", "")),
                    html.Td(a.get("measure", "—")),
                    html.Td(a.get("channel", "—")),
                    html.Td(a.get("variant", "—")),
                    html.Td(status),
                    html.Td(f"{a.get('reward', 0):.3f}"),
                ]))
            table = html.Table([
                html.Thead(html.Tr([
                    html.Th("Day"), html.Th("Patient"), html.Th("Measure"),
                    html.Th("Channel"), html.Th("Variant"), html.Th("Engagement"), html.Th("Reward"),
                ])),
                html.Tbody(rows),
            ], style={"width": "100%", "fontSize": "12px"})
        else:
            table = html.P("No actions yet...")

        # Action by channel
        if actions:
            channel_counts = Counter(a.get("channel", "none") for a in actions if a.get("action_id", 0) != 0)
            ch_fig = go.Figure(go.Bar(x=list(channel_counts.keys()), y=list(channel_counts.values())))
            ch_fig.update_layout(title="Actions by Channel", margin=dict(l=40, r=20, t=50, b=40))
        else:
            ch_fig = go.Figure()

        # Action by measure
        if actions:
            measure_counts = Counter(a.get("measure", "none") for a in actions if a.get("action_id", 0) != 0)
            top_measures = dict(measure_counts.most_common(10))
            m_fig = go.Figure(go.Bar(x=list(top_measures.keys()), y=list(top_measures.values())))
            m_fig.update_layout(title="Top Measures Targeted", margin=dict(l=40, r=20, t=50, b=40))
        else:
            m_fig = go.Figure()

        # Action vs no-action
        if actions:
            action_count = sum(1 for a in actions if a.get("action_id", 0) != 0)
            noaction_count = sum(1 for a in actions if a.get("action_id", 0) == 0)
            pie = go.Figure(go.Pie(labels=["Action", "No Action"], values=[action_count, noaction_count]))
            pie.update_layout(title="Action vs No-Action", margin=dict(l=20, r=20, t=50, b=20))
        else:
            pie = go.Figure()

        # --- Animated Bubble Chart: Patient cohort activity ---
        if actions:
            # Aggregate per-patient stats from latest day
            from collections import defaultdict
            patient_stats = defaultdict(lambda: {"actions": 0, "reward": 0, "measures": set(), "channels": set()})
            for a in actions:
                pid = a.get("patient_id", "")
                if a.get("action_id", 0) != 0:
                    patient_stats[pid]["actions"] += 1
                    patient_stats[pid]["reward"] += a.get("reward", 0)
                    if a.get("measure"):
                        patient_stats[pid]["measures"].add(a["measure"])
                    if a.get("channel"):
                        patient_stats[pid]["channels"].add(a["channel"])

            # Sample up to 200 patients for visualization
            sampled = list(patient_stats.items())[:200]
            if sampled:
                import random
                bubble = go.Figure()
                colors = [HUMANA_GREEN, NAVY, "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
                for i, (pid, stats) in enumerate(sampled):
                    primary_measure = list(stats["measures"])[0] if stats["measures"] else "none"
                    bubble.add_trace(go.Scatter(
                        x=[stats["actions"] + random.gauss(0, 0.3)],
                        y=[stats["reward"]],
                        mode="markers",
                        marker=dict(
                            size=max(8, min(stats["actions"] * 4, 30)),
                            color=colors[hash(primary_measure) % len(colors)],
                            opacity=0.6,
                            line=dict(width=1, color="white"),
                        ),
                        text=f"{pid}: {stats['actions']} actions, reward={stats['reward']:.2f}",
                        hoverinfo="text",
                        showlegend=False,
                    ))
                _styled_fig(bubble)
                bubble.update_layout(
                    title="Patient Activity Distribution",
                    xaxis_title="Total Actions", yaxis_title="Cumulative Reward",
                )
            else:
                bubble = go.Figure()
                _styled_fig(bubble)
        else:
            bubble = _empty_fig("Patient Cohort Activity — waiting for data...")

        return bubble, table, ch_fig, m_fig, pie

    # =========================================================================
    # Tab 3: Training Performance
    # =========================================================================
    @app.callback(
        [Output("champion-challenger", "figure"),
         Output("model-version-timeline", "figure"),
         Output("promotion-history-table", "children")],
        Input("interval-component", "n_intervals"),
    )
    def update_training(_):
        metrics = load_cumulative_metrics()

        # Champion vs Challenger
        if metrics:
            days = [m["day"] for m in metrics if m.get("champion_score") is not None]
            champ = [m["champion_score"] for m in metrics if m.get("champion_score") is not None]
            chall = [m["challenger_score"] for m in metrics if m.get("challenger_score") is not None]
            cc_fig = go.Figure()
            cc_fig.add_trace(go.Scatter(x=days, y=champ, mode="lines+markers", name="Champion"))
            cc_fig.add_trace(go.Scatter(x=days, y=chall, mode="lines+markers", name="Challenger"))
            cc_fig.update_layout(title="Champion vs Challenger", xaxis_title="Day",
                               yaxis_title="Mean Reward", margin=dict(l=40, r=20, t=50, b=40))
        else:
            cc_fig = _empty_fig("Champion vs Challenger — waiting for data...")

        # Model version timeline
        if metrics:
            days = [m["day"] for m in metrics]
            versions = [m.get("model_version", 1) for m in metrics]
            mv_fig = go.Figure(go.Scatter(x=days, y=versions, mode="lines+markers"))
            mv_fig.update_layout(title="Model Version", xaxis_title="Day", yaxis_title="Version",
                               margin=dict(l=40, r=20, t=50, b=40))
        else:
            mv_fig = go.Figure()

        # Promotion history
        if metrics:
            rows = []
            for m in metrics:
                if m.get("model_promoted"):
                    rows.append(html.Tr([
                        html.Td(f"Day {m['day']}"),
                        html.Td(f"v{m.get('model_version', '?')}"),
                        html.Td(f"{m.get('champion_score', 0):.4f}"),
                        html.Td(f"{m.get('challenger_score', 0):.4f}"),
                    ], style={"backgroundColor": "#e8f5e9"}))
            if rows:
                table = html.Table([
                    html.Thead(html.Tr([
                        html.Th("Day"), html.Th("New Version"), html.Th("Old Score"), html.Th("New Score"),
                    ])),
                    html.Tbody(rows),
                ], style={"width": "100%"})
            else:
                table = html.P("No promotions yet.")
        else:
            table = html.P("Waiting for training data...")

        return cc_fig, mv_fig, table

    # =========================================================================
    # Tab 4: Measure Deep Dive — Channel × Measure Chord/Heatmap
    # =========================================================================
    @app.callback(
        Output("channel-measure-chord", "figure"),
        Input("interval-component", "n_intervals"),
    )
    def update_chord(_):
        actions = load_all_actions()
        if not actions:
            fig = _empty_fig("Channel × Measure Effectiveness — waiting for data...")
            return fig

        # Build channel × measure click rate matrix
        from config import CHANNELS
        from collections import defaultdict
        counts = defaultdict(lambda: {"total": 0, "clicked": 0})
        for a in actions:
            if a.get("action_id", 0) == 0:
                continue
            ch = a.get("channel", "unknown")
            m = a.get("measure", "unknown")
            eng = a.get("engagement", {})
            key = (ch, m)
            counts[key]["total"] += 1
            if eng.get("clicked"):
                counts[key]["clicked"] += 1

        # Get measures that actually appear
        active_measures = sorted(set(a.get("measure") for a in actions if a.get("action_id", 0) != 0 and a.get("measure")))
        if not active_measures:
            active_measures = HEDIS_MEASURES[:6]
        active_channels = [c for c in CHANNELS if any(counts[(c, m)]["total"] > 0 for m in active_measures)]
        if not active_channels:
            active_channels = CHANNELS

        z = []
        text = []
        for ch in active_channels:
            row = []
            text_row = []
            for m in active_measures:
                c = counts[(ch, m)]
                rate = c["clicked"] / max(c["total"], 1)
                row.append(rate)
                text_row.append(f"{ch}×{m}<br>Click rate: {rate:.1%}<br>Total: {c['total']}")
            z.append(row)
            text.append(text_row)

        fig = go.Figure(go.Heatmap(
            z=z, x=active_measures, y=[c.upper() for c in active_channels],
            text=text, hoverinfo="text",
            colorscale=[[0, "#f8fafc"], [0.2, "#bae6fd"], [0.5, "#38bdf8"], [0.8, "#0284c7"], [1.0, NAVY]],
            colorbar=dict(title="Click Rate", thickness=15),
        ))
        _styled_fig(fig)
        fig.update_layout(
            title="Channel × Measure Click-Through Rate",
            xaxis_title="", yaxis_title="",
            yaxis=dict(dtick=1),
        )
        return fig

    @app.callback(
        [Output("measure-closure-trend", "figure"),
         Output("measure-channel-effectiveness", "figure"),
         Output("measure-funnel", "figure")],
        [Input("interval-component", "n_intervals"),
         Input("measure-selector", "value")],
    )
    def update_measures(_, selected_measure):
        metrics = load_cumulative_metrics()
        actions = load_all_actions()

        if not selected_measure:
            selected_measure = "COL"

        # Closure trend for selected measure
        if metrics:
            days = [m["day"] for m in metrics]
            rates = [m.get("measure_closure_rates", {}).get(selected_measure, 0) for m in metrics]
            trend = go.Figure()
            trend.add_trace(go.Scatter(x=days, y=rates, mode="lines+markers", name="Closure Rate"))
            trend.add_hline(y=0.68, line_dash="dash", line_color="green", annotation_text="4-Star Cut Point")
            trend.update_layout(
                title=f"{selected_measure} Gap Closure Rate",
                xaxis_title="Day", yaxis_title="Closure Rate",
                yaxis=dict(range=[0, 1]), margin=dict(l=40, r=20, t=50, b=40),
            )
        else:
            trend = go.Figure()

        # Channel effectiveness for this measure
        measure_actions = [a for a in actions if a.get("measure") == selected_measure and a.get("action_id", 0) != 0]
        if measure_actions:
            channel_engagement = {}
            for a in measure_actions:
                ch = a.get("channel", "unknown")
                eng = a.get("engagement", {})
                if ch not in channel_engagement:
                    channel_engagement[ch] = {"total": 0, "delivered": 0, "opened": 0, "clicked": 0}
                channel_engagement[ch]["total"] += 1
                if eng.get("delivered"):
                    channel_engagement[ch]["delivered"] += 1
                if eng.get("opened"):
                    channel_engagement[ch]["opened"] += 1
                if eng.get("clicked"):
                    channel_engagement[ch]["clicked"] += 1

            channels = list(channel_engagement.keys())
            click_rates = [channel_engagement[c]["clicked"] / max(channel_engagement[c]["total"], 1) for c in channels]
            eff = go.Figure(go.Bar(x=channels, y=click_rates))
            eff.update_layout(title=f"Channel Click Rate for {selected_measure}",
                            yaxis_title="Click Rate", margin=dict(l=40, r=20, t=50, b=40))
        else:
            eff = go.Figure()
            eff.update_layout(title=f"Channel Effectiveness for {selected_measure} (no data)")

        # Funnel
        if measure_actions:
            total = len(measure_actions)
            delivered = sum(1 for a in measure_actions if a.get("engagement", {}).get("delivered"))
            opened = sum(1 for a in measure_actions if a.get("engagement", {}).get("opened"))
            clicked = sum(1 for a in measure_actions if a.get("engagement", {}).get("clicked"))
            funnel = go.Figure(go.Funnel(
                y=["Sent", "Delivered", "Viewed", "Clicked"],
                x=[total, delivered, opened, clicked],
            ))
            funnel.update_layout(title=f"Patient Funnel: {selected_measure}",
                               margin=dict(l=40, r=20, t=50, b=40))
        else:
            funnel = go.Figure()

        return trend, eff, funnel

    # =========================================================================
    # Tab 5: Patient Journey
    # =========================================================================
    @app.callback(
        Output("patient-selector", "options"),
        Input("interval-component", "n_intervals"),
    )
    def update_patient_list(_):
        patient_ids = get_all_patient_ids()
        return [{"label": pid, "value": pid} for pid in patient_ids[:200]]

    @app.callback(
        [Output("patient-summary", "children"),
         Output("patient-budget-bar", "children"),
         Output("patient-action-cards", "children"),
         Output("patient-reward-curve", "figure"),
         Output("patient-gap-status", "figure")],
        [Input("patient-selector", "value"),
         Input("interval-component", "n_intervals")],
    )
    def update_patient_journey(patient_id, _):
        empty_fig = go.Figure()
        empty_fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))

        if not patient_id:
            return html.P("Select a patient to view their journey."), "", [], empty_fig, empty_fig

        journey = get_patient_journey(patient_id)
        if not journey:
            return html.P(f"No data for {patient_id}"), "", [], empty_fig, empty_fig

        # --- Summary ---
        total_actions = sum(1 for a in journey if a.get("action_id", 0) != 0)
        no_actions = sum(1 for a in journey if a.get("action_id", 0) == 0)
        total_reward = sum(a.get("reward", 0) for a in journey)
        days_active = len(set(a.get("day") for a in journey))
        summary = html.Div([
            html.H4(f"Patient: {patient_id}", style={"marginBottom": "5px"}),
            html.Div([
                html.Span(f"Actions: {total_actions}", style={"marginRight": "20px", "fontWeight": "bold"}),
                html.Span(f"Skipped: {no_actions}", style={"marginRight": "20px", "color": "#888"}),
                html.Span(f"Days: {days_active}", style={"marginRight": "20px"}),
                html.Span(f"Total Reward: {total_reward:.3f}", style={"fontWeight": "bold", "color": "#2ca02c" if total_reward > 0 else "#d62728"}),
            ], style={"fontSize": "14px"}),
        ])

        # --- Budget Bar ---
        latest = journey[-1] if journey else {}
        budget_rem = latest.get("budget_remaining", 12)
        budget_max = latest.get("budget_max", 12)
        budget_pct = (budget_rem / max(budget_max, 1)) * 100
        budget_color = "#4caf50" if budget_pct > 50 else ("#ff9800" if budget_pct > 25 else "#f44336")
        budget_bar = html.Div([
            html.Div([
                html.Span("Message Budget: ", style={"fontWeight": "bold"}),
                html.Span(f"{budget_rem}/{budget_max} remaining"),
            ], style={"marginBottom": "5px"}),
            html.Div([
                html.Div(style={
                    "width": f"{budget_pct}%", "height": "20px",
                    "backgroundColor": budget_color, "borderRadius": "4px",
                    "transition": "width 0.5s",
                }),
            ], style={
                "width": "100%", "backgroundColor": "#e0e0e0",
                "borderRadius": "4px", "overflow": "hidden",
            }),
        ])

        # --- Action Cards ---
        channel_icons = {"sms": "📱", "email": "📧", "portal": "🌐", "app": "📲", "ivr": "📞"}
        channel_colors = {"sms": "#e3f2fd", "email": "#fff3e0", "portal": "#e8f5e9",
                         "app": "#fce4ec", "ivr": "#f3e5f5"}

        cards = []
        for a in journey:
            action_id = a.get("action_id", 0)
            eng = a.get("engagement", {})
            day = a.get("day", "?")
            measure = a.get("measure")
            channel = a.get("channel")
            variant = a.get("variant", "")
            reward = a.get("reward", 0)

            if action_id == 0:
                # No-action card (smaller, greyed out)
                cards.append(html.Div([
                    html.Div([
                        html.Span(f"Day {day}", style={"fontWeight": "bold", "fontSize": "11px"}),
                        html.Span(" ⏸️ No Action", style={"fontSize": "11px", "color": "#888"}),
                    ]),
                ], style={
                    "border": "1px dashed #ccc", "borderRadius": "8px", "padding": "8px 12px",
                    "backgroundColor": "#fafafa", "minWidth": "120px", "opacity": "0.6",
                }))
                continue

            # Determine disposition
            if eng.get("clicked") or eng.get("completed"):
                disposition_icon = "👍"
                disposition_text = "Clicked"
                disposition_color = "#4caf50"
                card_border = "2px solid #4caf50"
            elif eng.get("opened"):
                disposition_icon = "👁️"
                disposition_text = "Viewed"
                disposition_color = "#ff9800"
                card_border = "2px solid #ff9800"
            elif eng.get("delivered"):
                disposition_icon = "📬"
                disposition_text = "Delivered"
                disposition_color = "#2196f3"
                card_border = "1px solid #2196f3"
            elif eng.get("failed"):
                disposition_icon = "👎"
                disposition_text = "Failed"
                disposition_color = "#f44336"
                card_border = "2px solid #f44336"
            elif eng.get("expired"):
                disposition_icon = "⏰"
                disposition_text = "Expired"
                disposition_color = "#9e9e9e"
                card_border = "1px solid #9e9e9e"
            else:
                disposition_icon = "📤"
                disposition_text = "Sent"
                disposition_color = "#666"
                card_border = "1px solid #ddd"

            ch_icon = channel_icons.get(channel, "❓")
            ch_bg = channel_colors.get(channel, "#f5f5f5")

            # Variant display name
            variant_display = (variant or "").replace("_", " ").title()

            card = html.Div([
                # Header: Day + Channel icon
                html.Div([
                    html.Span(f"Day {day}", style={"fontWeight": "bold", "fontSize": "12px"}),
                    html.Span(f" {ch_icon} {(channel or '').upper()}", style={
                        "fontSize": "12px", "marginLeft": "8px", "color": "#555",
                    }),
                ], style={"marginBottom": "6px", "borderBottom": "1px solid #eee", "paddingBottom": "4px"}),

                # Measure badge
                html.Div([
                    html.Span(measure or "—", style={
                        "backgroundColor": NAVY, "color": "white", "padding": "2px 8px",
                        "borderRadius": "12px", "fontSize": "11px", "fontWeight": "bold",
                    }),
                ], style={"marginBottom": "6px"}),

                # Variant description
                html.Div(variant_display, style={
                    "fontSize": "11px", "color": "#555", "marginBottom": "8px",
                    "lineHeight": "1.3",
                }),

                # Disposition row: icon + text + reward
                html.Div([
                    html.Span(f"{disposition_icon} ", style={"fontSize": "18px"}),
                    html.Span(disposition_text, style={
                        "color": disposition_color, "fontWeight": "bold", "fontSize": "12px",
                    }),
                    html.Span(f"  +{reward:.3f}", style={
                        "fontSize": "11px", "color": "#888", "marginLeft": "auto",
                    }),
                ], style={"display": "flex", "alignItems": "center"}),

            ], style={
                "border": card_border, "borderRadius": "10px", "padding": "10px 14px",
                "backgroundColor": ch_bg, "minWidth": "180px", "maxWidth": "220px",
                "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
            })
            cards.append(card)

        # --- Reward Curve ---
        rewards = [a.get("reward", 0) for a in journey]
        cum_rewards = []
        total = 0
        for r in rewards:
            total += r
            cum_rewards.append(total)
        reward_fig = go.Figure()
        reward_fig.add_trace(go.Scatter(y=cum_rewards, mode="lines+markers", name="Cumulative Reward",
                                        line=dict(color="#2ca02c")))
        reward_fig.update_layout(title="Cumulative Reward", xaxis_title="Interaction",
                               yaxis_title="Reward", margin=dict(l=40, r=20, t=50, b=40))

        # --- Actions by Measure ---
        measure_counts = Counter(a.get("measure") for a in journey if a.get("action_id", 0) != 0)
        gap_fig = go.Figure(go.Bar(
            x=list(measure_counts.keys()), y=list(measure_counts.values()),
            marker_color="#1565c0",
        ))
        gap_fig.update_layout(title="Actions by Measure", margin=dict(l=40, r=20, t=50, b=40))

        return summary, budget_bar, cards, reward_fig, gap_fig

    # =========================================================================
    # Tab 6: Action Lifecycle — Sankey Flow Diagram
    # =========================================================================
    @app.callback(
        Output("sm-sankey", "figure"),
        Input("interval-component", "n_intervals"),
    )
    def update_sankey(_):
        sm_data = load_all_state_machine_data()
        if not sm_data:
            fig = _empty_fig("Action Flow Sankey — waiting for data...")
            return fig

        # Count transitions between states
        from collections import defaultdict
        transition_counts = defaultdict(int)
        for record in sm_data:
            history = record.get("state_history", [])
            for i in range(len(history) - 1):
                src = history[i]["state"]
                dst = history[i + 1]["state"]
                transition_counts[(src, dst)] += 1

        if not transition_counts:
            fig = _empty_fig("No transitions recorded yet...")
            return fig

        # Build Sankey nodes and links
        all_states = ["CREATED", "QUEUED", "PRESENTED", "VIEWED",
                      "ACCEPTED", "COMPLETED", "DECLINED", "FAILED", "EXPIRED"]
        state_colors = {
            "CREATED": "#94a3b8", "QUEUED": "#64748b",
            "PRESENTED": "#3b82f6", "VIEWED": "#8b5cf6",
            "ACCEPTED": HUMANA_GREEN, "COMPLETED": "#15803d",
            "DECLINED": "#f59e0b", "FAILED": "#ef4444", "EXPIRED": "#9ca3af",
        }
        node_indices = {s: i for i, s in enumerate(all_states)}

        sources, targets, values, link_colors = [], [], [], []
        for (src, dst), count in transition_counts.items():
            if src in node_indices and dst in node_indices:
                sources.append(node_indices[src])
                targets.append(node_indices[dst])
                values.append(count)
                # Color link by destination
                c = state_colors.get(dst, "#94a3b8")
                link_colors.append(c.replace("#", "rgba(") if not c.startswith("rgba") else c)

        # Convert hex to rgba for links
        def hex_to_rgba(hex_color, alpha=0.4):
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        link_colors_rgba = [hex_to_rgba(state_colors.get(all_states[t], "#94a3b8"), 0.4)
                           for t in targets]

        fig = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                pad=20, thickness=25,
                label=all_states,
                color=[state_colors.get(s, "#94a3b8") for s in all_states],
                line=dict(color="white", width=1),
            ),
            link=dict(
                source=sources, target=targets, value=values,
                color=link_colors_rgba,
            ),
        ))
        _styled_fig(fig)
        fig.update_layout(
            title="Action Lifecycle Flow",
            font=dict(size=13),
        )
        return fig

    # =========================================================================
    # Tab 6: Action Lifecycle State Machine — Funnel & Conversions
    # =========================================================================
    @app.callback(
        [Output("sm-funnel", "figure"),
         Output("sm-channel-funnel", "figure"),
         Output("sm-transitions-table", "children"),
         Output("sm-conversion-rates", "figure")],
        Input("interval-component", "n_intervals"),
    )
    def update_state_machine(_):
        sm_data = load_all_state_machine_data()

        if not sm_data:
            empty = go.Figure()
            empty.update_layout(title="Waiting for state machine data...")
            return empty, empty, html.P("No data yet"), empty

        # Overall funnel
        state_counts = Counter(r.get("current_state", "UNKNOWN") for r in sm_data)
        ordered_states = ["CREATED", "QUEUED", "PRESENTED", "VIEWED", "ACCEPTED", "COMPLETED", "DECLINED", "FAILED", "EXPIRED"]
        funnel_vals = [state_counts.get(s, 0) for s in ordered_states]
        funnel = go.Figure(go.Funnel(
            y=ordered_states, x=funnel_vals,
            textinfo="value+percent initial",
        ))
        funnel.update_layout(title="Action Lifecycle Funnel", margin=dict(l=100, r=20, t=50, b=40))

        # Channel funnel
        channel_states: dict = {}
        for r in sm_data:
            ch = r.get("channel", "unknown")
            state = r.get("current_state", "UNKNOWN")
            if ch not in channel_states:
                channel_states[ch] = Counter()
            channel_states[ch][state] += 1

        ch_funnel = go.Figure()
        for ch, counts in channel_states.items():
            presented = counts.get("PRESENTED", 0) + counts.get("VIEWED", 0) + counts.get("ACCEPTED", 0) + counts.get("COMPLETED", 0) + counts.get("DECLINED", 0)
            viewed = counts.get("VIEWED", 0) + counts.get("ACCEPTED", 0) + counts.get("COMPLETED", 0) + counts.get("DECLINED", 0)
            accepted = counts.get("ACCEPTED", 0) + counts.get("COMPLETED", 0)
            completed = counts.get("COMPLETED", 0)
            total = sum(counts.values())
            ch_funnel.add_trace(go.Bar(
                name=ch,
                x=["Presented", "Viewed", "Accepted", "Completed"],
                y=[presented / max(total, 1), viewed / max(total, 1),
                   accepted / max(total, 1), completed / max(total, 1)],
            ))
        ch_funnel.update_layout(title="Conversion by Channel", barmode="group",
                               yaxis_title="Rate", margin=dict(l=40, r=20, t=50, b=40))

        # Recent transitions table
        recent = sm_data[-30:]
        rows = []
        for r in reversed(recent):
            history = r.get("state_history", [])
            current = r.get("current_state", "?")
            rows.append(html.Tr([
                html.Td(r.get("tracking_id", "")[:25]),
                html.Td(r.get("patient_id", "")),
                html.Td(r.get("measure", "")),
                html.Td(r.get("channel", "")),
                html.Td(current, style={
                    "color": "green" if current == "COMPLETED" else ("red" if current in ("FAILED", "EXPIRED") else "black")
                }),
                html.Td(f"Day {r.get('day_created', '?')}"),
            ]))
        table = html.Table([
            html.Thead(html.Tr([
                html.Th("Tracking ID"), html.Th("Patient"), html.Th("Measure"),
                html.Th("Channel"), html.Th("State"), html.Th("Created"),
            ])),
            html.Tbody(rows),
        ], style={"width": "100%", "fontSize": "12px"})

        # Conversion rates
        total = len(sm_data)
        completed = state_counts.get("COMPLETED", 0)
        failed = state_counts.get("FAILED", 0)
        expired = state_counts.get("EXPIRED", 0)
        declined = state_counts.get("DECLINED", 0)
        conv = go.Figure(go.Bar(
            x=["Completed", "Declined", "Failed", "Expired"],
            y=[completed / max(total, 1), declined / max(total, 1),
               failed / max(total, 1), expired / max(total, 1)],
            marker_color=["green", "orange", "red", "grey"],
        ))
        conv.update_layout(title="Terminal State Distribution", yaxis_title="Rate",
                         margin=dict(l=40, r=20, t=50, b=40))

        return funnel, ch_funnel, table, conv

    # =========================================================================
    # Tab 7: Simulation Logs
    # =========================================================================
    @app.callback(
        Output("log-stream", "children"),
        [Input("interval-component", "n_intervals"),
         Input("log-level-filter", "value")],
    )
    def update_logs(_, level_filter):
        logs = load_simulation_logs(max_lines=200)

        if not logs:
            return html.Span("Waiting for simulation to start...",
                           style={"color": "#888"})

        if level_filter and level_filter != "ALL":
            logs = [l for l in logs if l.get("level") == level_filter]

        lines = []
        color_map = {
            "PHASE": "#569cd6",   # Blue
            "METRIC": "#4ec9b0",  # Teal
            "INFO": "#d4d4d4",    # Grey
            "ERROR": "#f44747",   # Red
        }
        for entry in logs:
            level = entry.get("level", "INFO")
            ts = entry.get("timestamp", "")[:19]  # Trim microseconds
            msg = entry.get("message", "")
            color = color_map.get(level, "#d4d4d4")
            weight = "bold" if level in ("PHASE", "METRIC") else "normal"

            lines.append(html.Div(
                f"[{ts}] [{level:6s}] {msg}",
                style={"color": color, "fontWeight": weight,
                       "borderBottom": "1px solid #333" if level == "PHASE" else "none",
                       "paddingBottom": "2px" if level == "PHASE" else "0",
                       "marginBottom": "2px" if level == "PHASE" else "0"},
            ))

        return lines
