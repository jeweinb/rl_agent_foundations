"""
Dash callbacks for all dashboard tabs.
Handles real-time data updates via dcc.Interval polling.
"""
from dash import Input, Output, State, callback, html, dash_table
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter

from config import HEDIS_MEASURES, MEASURE_DESCRIPTIONS, MEASURE_WEIGHTS, STARS_BONUS_THRESHOLD
from dashboard.theme import (
    HUMANA_GREEN, HUMANA_DARK_GREEN, NAVY, NAVY_LIGHT,
    GRAY_BG, GRAY_BORDER, WHITE, GRAY_TEXT,
    STATUS_SUCCESS, STATUS_SUCCESS_BG, STATUS_WARNING, STATUS_WARNING_BG,
    STATUS_ERROR, STATUS_ERROR_BG, STATE_COLORS,
    CHART_TEMPLATE, CHART_FONT, CHART_MARGIN,
    CHANNEL_COLORS, CHANNEL_BG_COLORS, CHANNEL_ICONS,
)

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
                   "suffix": " to 4★", "valueformat": "+.2f"},
            number={"valueformat": ".2f"},
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
        [Output("global-budget-gauge", "children"),
         Output("action-leaderboard", "figure"),
         Output("recent-actions-table", "children"),
         Output("action-by-channel", "figure"),
         Output("action-by-measure", "figure"),
         Output("action-vs-noaction", "figure")],
        [Input("interval-component", "n_intervals"),
         Input("leaderboard-rank-by", "value")],
    )
    def update_realtime(_, rank_by):
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

        # --- Global Budget Gauge ---
        budget_gauge = html.P("Waiting for budget data...", style={"color": GRAY_TEXT})
        metrics = load_cumulative_metrics()
        if metrics:
            latest = metrics[-1]
            budget_remaining = latest.get("avg_budget_remaining")  # Now stores global remaining
            from config import compute_global_budget, COHORT_SIZE
            budget_total = compute_global_budget(COHORT_SIZE)
            if budget_remaining is not None:
                budget_used = budget_total - budget_remaining
                budget_pct = max(0, budget_remaining / max(budget_total, 1) * 100)
                used_pct = 100 - budget_pct
                bar_color = HUMANA_GREEN if budget_pct > 50 else ("#f59e0b" if budget_pct > 25 else "#ef4444")
                budget_gauge = html.Div([
                    html.Div([
                        html.Div([
                            html.Span(f"{budget_remaining:,.0f}", style={"fontSize": "28px", "fontWeight": "700", "color": NAVY}),
                            html.Span(f" / {budget_total:,} messages remaining", style={"fontSize": "14px", "color": GRAY_TEXT}),
                        ]),
                        html.Div([
                            html.Span(f"{budget_used:,.0f}", style={"fontSize": "28px", "fontWeight": "700", "color": "#64748b"}),
                            html.Span(f" used ({used_pct:.0f}%)", style={"fontSize": "14px", "color": GRAY_TEXT}),
                        ]),
                    ], style={"display": "flex", "gap": "60px", "marginBottom": "10px"}),
                    html.Div([
                        html.Div(style={
                            "width": f"{used_pct:.0f}%", "height": "14px",
                            "backgroundColor": "#94a3b8",
                        }),
                        html.Div(style={
                            "width": f"{budget_pct:.0f}%", "height": "14px",
                            "backgroundColor": bar_color,
                        }),
                    ], style={"display": "flex", "borderRadius": "7px", "overflow": "hidden",
                             "border": f"1px solid {GRAY_BORDER}"}),
                ])

        # --- Action Leaderboard with ranking toggle ---
        if not rank_by:
            rank_by = "q_value"

        sm_data = load_all_state_machine_data()

        if rank_by == "q_value":
            # Use the trained model's Q-values to rank actions
            import torch
            import numpy as np
            from config import NUM_ACTIONS, STATE_DIM, CHECKPOINTS_DIR, ACTION_BY_ID
            import os

            champion_path = os.path.join(CHECKPOINTS_DIR, "champion.pt")
            if os.path.exists(champion_path):
                from training.cql_trainer import ActorCriticCQL
                agent = ActorCriticCQL()
                try:
                    agent.load_state_dict(torch.load(champion_path, weights_only=True))
                except Exception:
                    agent = ActorCriticCQL()

                # Get average Q-value per action across a sample of states
                agent.critic.eval()
                with torch.no_grad():
                    # Use random states as a representative sample
                    sample_states = torch.randn(100, STATE_DIM)
                    q_min = agent.critic.q_min(sample_states)  # (100, NUM_ACTIONS)
                    avg_q = q_min.mean(dim=0).numpy()  # (NUM_ACTIONS,)

                # Build leaderboard from Q-values
                action_data = []
                for aid in range(1, NUM_ACTIONS):  # Skip no_action
                    act = ACTION_BY_ID.get(aid)
                    if act:
                        label = f"{act.measure} | {act.channel.upper()} | {act.variant.replace('_', ' ').title()}"
                        action_data.append((label, float(avg_q[aid]), act.measure))

                action_data.sort(key=lambda x: x[1], reverse=True)
                top = action_data[:20]

                if top:
                    labels = [t[0] for t in top]
                    values = [t[1] for t in top]
                    colors = [HUMANA_GREEN if v > 0 else "#94a3b8" for v in values]

                    leaderboard = go.Figure(go.Bar(
                        y=labels, x=values, orientation="h",
                        marker_color=colors,
                        text=[f"{v:.3f}" for v in values],
                        textposition="outside",
                    ))
                    _styled_fig(leaderboard)
                    leaderboard.update_layout(
                        title="Top 20 Actions by Q-Value (Model's Predicted Future Reward)",
                        xaxis_title="Average Q-Value",
                        yaxis=dict(autorange="reversed"),
                    )
                else:
                    leaderboard = _empty_fig("No model checkpoint found")
            else:
                leaderboard = _empty_fig("Model not yet trained — Q-values unavailable")

        elif sm_data:
            # Rank by acceptance or completion from state machine
            from collections import defaultdict
            action_perf = defaultdict(lambda: {"total": 0, "accepted": 0, "completed": 0})
            for r in sm_data:
                label = f"{r.get('measure', '?')} | {r.get('channel', '?').upper()} | {r.get('variant', '?').replace('_', ' ').title()}"
                state = r.get("current_state", "")
                action_perf[label]["total"] += 1
                if state in ("ACCEPTED", "COMPLETED"):
                    action_perf[label]["accepted"] += 1
                if state == "COMPLETED":
                    action_perf[label]["completed"] += 1

            qualified = {k: v for k, v in action_perf.items() if v["total"] >= 3}
            if qualified:
                metric_key = "completed" if rank_by == "completion" else "accepted"
                metric_label = "Completion Rate" if rank_by == "completion" else "Acceptance Rate"

                sorted_actions = sorted(qualified.items(),
                                       key=lambda x: x[1][metric_key] / max(x[1]["total"], 1),
                                       reverse=True)[:20]
                labels = [k for k, _ in sorted_actions]
                rates = [v[metric_key] / max(v["total"], 1) for _, v in sorted_actions]
                hover = [
                    f"{k}<br>Accepted: {v['accepted']}/{v['total']} ({v['accepted']/max(v['total'],1):.0%})"
                    f"<br>Completed: {v['completed']}/{v['total']} ({v['completed']/max(v['total'],1):.0%})"
                    for k, v in sorted_actions
                ]
                colors = [HUMANA_GREEN if r > 0.10 else (NAVY if r > 0.03 else "#94a3b8") for r in rates]

                leaderboard = go.Figure(go.Bar(
                    y=labels, x=rates, orientation="h",
                    marker_color=colors,
                    text=[f"{r:.0%}" for r in rates],
                    textposition="outside",
                    hovertext=hover, hoverinfo="text",
                ))
                _styled_fig(leaderboard)
                leaderboard.update_layout(
                    title=f"Top 20 Actions by {metric_label}",
                    xaxis_title=metric_label,
                    yaxis=dict(autorange="reversed"),
                )
            else:
                leaderboard = _empty_fig("Not enough data for leaderboard...")
        else:
            leaderboard = _empty_fig("Action Leaderboard — waiting for data...")

        return budget_gauge, leaderboard, table, ch_fig, m_fig, pie

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
        # Use state machine data for accurate lifecycle stats (not stale action records)
        sm_data = load_all_state_machine_data()
        if not sm_data:
            fig = _empty_fig("Channel × Measure Effectiveness — waiting for data...")
            return fig

        from config import CHANNELS
        from collections import defaultdict

        # Count acceptance rate per channel × measure from state machine
        counts = defaultdict(lambda: {"total": 0, "accepted": 0, "completed": 0, "viewed": 0})
        for r in sm_data:
            ch = r.get("channel", "unknown")
            m = r.get("measure", "unknown")
            state = r.get("current_state", "")
            counts[(ch, m)]["total"] += 1
            if state in ("VIEWED", "ACCEPTED", "COMPLETED", "DECLINED"):
                counts[(ch, m)]["viewed"] += 1
            if state in ("ACCEPTED", "COMPLETED"):
                counts[(ch, m)]["accepted"] += 1
            if state == "COMPLETED":
                counts[(ch, m)]["completed"] += 1

        active_measures = sorted(set(r.get("measure") for r in sm_data if r.get("measure")))
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
                rate = c["accepted"] / max(c["total"], 1)
                row.append(rate)
                text_row.append(
                    f"{ch.upper()} × {m}<br>"
                    f"Acceptance: {rate:.1%}<br>"
                    f"Viewed: {c['viewed']}/{c['total']}<br>"
                    f"Accepted: {c['accepted']}/{c['total']}<br>"
                    f"Completed: {c['completed']}/{c['total']}"
                )
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

        # Closure trend for selected measure with actual CMS cut points
        from config import MEASURE_CUT_POINTS
        cuts = MEASURE_CUT_POINTS.get(selected_measure, {})
        cut_4star = cuts.get(4, 0.70)
        cut_5star = cuts.get(5, 0.85)

        if metrics:
            days = [m["day"] for m in metrics]
            rates = [m.get("measure_closure_rates", {}).get(selected_measure, 0) for m in metrics]
            trend = go.Figure()
            trend.add_trace(go.Scatter(x=days, y=rates, mode="lines+markers", name="Closure Rate",
                                       line=dict(color=HUMANA_GREEN, width=3),
                                       marker=dict(size=6)))
            trend.add_hline(y=cut_4star, line_dash="dash", line_color="#f59e0b",
                          annotation_text=f"4★ ({cut_4star:.0%})")
            trend.add_hline(y=cut_5star, line_dash="dot", line_color=HUMANA_GREEN,
                          annotation_text=f"5★ ({cut_5star:.0%})")
            _styled_fig(trend)
            trend.update_layout(
                title=f"{selected_measure} — {MEASURE_DESCRIPTIONS.get(selected_measure, '')}",
                xaxis_title="Day", yaxis_title="Closure Rate",
                yaxis=dict(range=[0, 1]),
            )
        else:
            trend = _empty_fig(f"{selected_measure} — waiting for data...")

        # Channel effectiveness and funnel — use state machine data for accurate lifecycle
        sm_data = load_all_state_machine_data()
        measure_sm = [r for r in sm_data if r.get("measure") == selected_measure]

        if measure_sm:
            # Channel effectiveness from state machine terminal states
            from collections import defaultdict
            ch_stats = defaultdict(lambda: {"total": 0, "completed": 0, "accepted": 0, "viewed": 0, "presented": 0})
            for r in measure_sm:
                ch = r.get("channel", "unknown")
                state = r.get("current_state", "")
                ch_stats[ch]["total"] += 1
                if state in ("PRESENTED", "VIEWED", "ACCEPTED", "COMPLETED", "DECLINED"):
                    ch_stats[ch]["presented"] += 1
                if state in ("VIEWED", "ACCEPTED", "COMPLETED", "DECLINED"):
                    ch_stats[ch]["viewed"] += 1
                if state in ("ACCEPTED", "COMPLETED"):
                    ch_stats[ch]["accepted"] += 1
                if state == "COMPLETED":
                    ch_stats[ch]["completed"] += 1

            channels = sorted(ch_stats.keys())
            accept_rates = [ch_stats[c]["accepted"] / max(ch_stats[c]["total"], 1) for c in channels]
            eff = go.Figure(go.Bar(
                x=[c.upper() for c in channels], y=accept_rates,
                marker_color=HUMANA_GREEN,
            ))
            _styled_fig(eff)
            eff.update_layout(title=f"Channel Acceptance Rate — {selected_measure}",
                            yaxis_title="Acceptance Rate")
        else:
            eff = _empty_fig(f"Channel Effectiveness — {selected_measure} (no data)")

        # Funnel from state machine
        if measure_sm:
            from collections import Counter
            state_counts = Counter(r.get("current_state") for r in measure_sm)
            total = len(measure_sm)
            presented = sum(state_counts.get(s, 0) for s in ["PRESENTED", "VIEWED", "ACCEPTED", "COMPLETED", "DECLINED"])
            viewed = sum(state_counts.get(s, 0) for s in ["VIEWED", "ACCEPTED", "COMPLETED", "DECLINED"])
            accepted = sum(state_counts.get(s, 0) for s in ["ACCEPTED", "COMPLETED"])
            completed = state_counts.get("COMPLETED", 0)
            funnel = go.Figure(go.Funnel(
                y=["Created", "Presented", "Viewed", "Accepted", "Completed"],
                x=[total, presented, viewed, accepted, completed],
                marker=dict(color=[NAVY, "#3b82f6", "#8b5cf6", HUMANA_GREEN, "#15803d"]),
            ))
            _styled_fig(funnel)
            funnel.update_layout(title=f"Action Lifecycle Funnel — {selected_measure}")
        else:
            funnel = _empty_fig(f"Funnel — {selected_measure} (no data)")

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

        # --- Patient Message Count + Global Budget Context ---
        latest = journey[-1] if journey else {}
        patient_msgs = latest.get("patient_messages", sum(1 for a in journey if a.get("action_id", 0) != 0))
        budget_rem = latest.get("budget_remaining", 60000)
        budget_max = latest.get("budget_max", 60000)
        from config import AVG_MESSAGES_PER_PATIENT
        budget_used = budget_max - budget_rem
        # Patient contact intensity vs cohort average
        above_avg = patient_msgs > AVG_MESSAGES_PER_PATIENT
        intensity_color = "#f59e0b" if above_avg else HUMANA_GREEN
        intensity_label = "Above Average" if above_avg else "Below Average"
        # Visual: bar showing this patient's messages vs the avg
        msg_pct = min(patient_msgs / max(AVG_MESSAGES_PER_PATIENT * 2, 1) * 100, 100)
        avg_marker_pct = min(AVG_MESSAGES_PER_PATIENT / max(AVG_MESSAGES_PER_PATIENT * 2, 1) * 100, 100)

        budget_bar = html.Div([
            html.Div([
                html.Span(f"This patient: ", style={"fontWeight": "500", "fontSize": "14px"}),
                html.Span(f"{patient_msgs} messages received", style={"fontSize": "14px", "fontWeight": "700", "color": NAVY}),
                html.Span(f"  (avg: {AVG_MESSAGES_PER_PATIENT})", style={"fontSize": "12px", "color": GRAY_TEXT, "marginLeft": "8px"}),
                html.Span(f"  {intensity_label}", style={
                    "fontSize": "11px", "fontWeight": "700", "color": intensity_color,
                    "marginLeft": "12px", "padding": "2px 8px",
                    "backgroundColor": f"{intensity_color}18", "borderRadius": "4px",
                }),
            ], style={"marginBottom": "6px"}),
            html.Div([
                html.Div(style={
                    "width": f"{msg_pct}%", "height": "12px",
                    "backgroundColor": intensity_color, "borderRadius": "6px 0 0 6px",
                }),
                html.Div(style={
                    "width": f"{100-msg_pct}%", "height": "12px",
                }),
            ], style={
                "width": "100%", "backgroundColor": "#f1f5f9", "position": "relative",
                "borderRadius": "6px", "overflow": "hidden",
                "border": f"1px solid {GRAY_BORDER}", "display": "flex",
            }),
            # Average marker line
            html.Div(
                html.Span("avg", style={"fontSize": "9px", "color": GRAY_TEXT}),
                style={
                    "position": "relative", "left": f"{avg_marker_pct}%",
                    "top": "-14px", "height": "0", "width": "0",
                    "borderLeft": "1px dashed #64748b",
                },
            ),
        ], style={
            "background": "white", "padding": "12px 16px", "borderRadius": "8px",
            "border": f"1px solid {GRAY_BORDER}",
        })

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

        # --- Actions by Measure (with full names) ---
        measure_counts = Counter(a.get("measure") for a in journey if a.get("action_id", 0) != 0)
        measures_sorted = sorted(measure_counts.keys(), key=lambda m: measure_counts[m], reverse=True)
        labels = [f"{m} — {MEASURE_DESCRIPTIONS.get(m, m)}" for m in measures_sorted]
        gap_fig = go.Figure(go.Bar(
            x=labels, y=[measure_counts[m] for m in measures_sorted],
            marker_color=NAVY,
        ))
        _styled_fig(gap_fig)
        gap_fig.update_layout(title="Actions by Measure")

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

        # Build Sankey nodes and links — use centralized state ordering
        from simulation.action_state_machine import ALL_STATES_ORDERED
        all_states = [s.value for s in ALL_STATES_ORDERED]
        node_indices = {s: i for i, s in enumerate(all_states)}

        sources, targets, values, link_colors = [], [], [], []
        def hex_to_rgba(hex_color, alpha=0.4):
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        for (src, dst), count in transition_counts.items():
            if src in node_indices and dst in node_indices:
                sources.append(node_indices[src])
                targets.append(node_indices[dst])
                values.append(count)

        link_colors_rgba = [hex_to_rgba(STATE_COLORS.get(all_states[t], "#94a3b8"), 0.4)
                           for t in targets]

        fig = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                pad=20, thickness=25,
                label=all_states,
                color=[STATE_COLORS.get(s, "#94a3b8") for s in all_states],
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

        # Overall funnel — cumulative "reached this stage or beyond"
        # Count how many actions reached each stage by looking at state_history
        total = len(sm_data)
        from simulation.action_state_machine import LIFECYCLE_STAGES
        stage_order = [s.value for s in LIFECYCLE_STAGES]
        reached = {s: 0 for s in stage_order}
        for r in sm_data:
            states_visited = {sh.get("state", sh) for sh in r.get("state_history", [])}
            # Also count current_state
            states_visited.add(r.get("current_state", ""))
            for s in stage_order:
                if s in states_visited:
                    reached[s] += 1

        funnel_labels = ["Created", "Queued", "Presented", "Viewed", "Accepted", "Completed"]
        funnel_vals = [reached[s] for s in stage_order]

        funnel = go.Figure(go.Funnel(
            y=funnel_labels, x=funnel_vals,
            textinfo="value+percent initial",
            marker=dict(color=[NAVY, "#475569", "#3b82f6", "#8b5cf6", HUMANA_GREEN, "#15803d"]),
        ))
        _styled_fig(funnel)
        funnel.update_layout(title="Action Lifecycle Funnel")

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
