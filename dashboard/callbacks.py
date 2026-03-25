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
    # Global: Day Counter in header
    # =========================================================================
    @app.callback(
        Output("day-counter", "children"),
        Input("interval-component", "n_intervals"),
    )
    def update_day_counter(_):
        metrics = load_cumulative_metrics()
        if not metrics:
            return html.Div([
                html.Span("Initializing...", style={"fontSize": "14px", "color": "#64748b"}),
            ])
        latest = metrics[-1]
        day = latest.get("day", 0)
        from config import SIMULATION_DAYS
        stars = latest.get("stars_score", 1.0)
        stars_color = "#00A664" if stars >= 4.0 else ("#f59e0b" if stars >= 3.0 else "#1B2A4A")
        return html.Div([
            html.Div([
                html.Span(f"Day {day}", style={"fontSize": "22px", "fontWeight": "700", "color": "#1B2A4A"}),
                html.Span(f" / {SIMULATION_DAYS}", style={"fontSize": "14px", "color": "#64748b"}),
            ]),
            html.Div([
                html.Span(f"STARS {stars:.2f}", style={"fontSize": "14px", "fontWeight": "600", "color": stars_color}),
            ]),
        ])

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

        # Regret Curve — oracle is the best single-day performance observed so far
        # As model improves, daily reward approaches oracle → regret flattens
        if metrics and len(metrics) > 1:
            reg = go.Figure()
            # Running best: track the best daily reward seen so far
            best_so_far = []
            running_best = float("-inf")
            for m in metrics:
                running_best = max(running_best, m["daily_reward"])
                best_so_far.append(running_best)

            # Regret = cumulative gap between current performance and running best
            regret = []
            cum_regret = 0.0
            for i, m in enumerate(metrics):
                cum_regret += best_so_far[i] - m["daily_reward"]
                regret.append(cum_regret)

            reg.add_trace(go.Scatter(x=days, y=regret, mode="lines", name="Cumulative Regret",
                                     line=dict(color=NAVY, width=2)))
            # Also show daily reward trend for context
            daily_rewards = [m["daily_reward"] for m in metrics]
            reg.add_trace(go.Scatter(x=days, y=daily_rewards, mode="lines+markers",
                                     name="Daily Reward", line=dict(color=HUMANA_GREEN, width=1, dash="dot"),
                                     yaxis="y2"))
            _styled_fig(reg)
            reg.update_layout(
                title="Regret (should flatten as model improves)",
                xaxis_title="Day",
                yaxis=dict(title="Cumulative Regret", side="left"),
                yaxis2=dict(title="Daily Reward", side="right", overlaying="y"),
                legend=dict(x=0.01, y=0.99),
            )
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
            hover_data = []
            for m in measures_with_data:
                row = [met.get("measure_closure_rates", {}).get(m, 0) for met in metrics]
                z_data.append(row)
                hover_data.append([f"{m} — {MEASURE_DESCRIPTIONS.get(m, m)}<br>Day {d}: {r:.1%}"
                                  for d, r in zip(days_list, row)])
            y_labels = [f"{m}" for m in measures_with_data]
            heatmap = go.Figure(go.Heatmap(
                z=z_data, x=days_list, y=y_labels,
                text=hover_data, hoverinfo="text",
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
         Output("action-vs-noaction", "figure"),
         Output("action-variant-breakdown", "children")],
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
            ch_fig = go.Figure(go.Bar(
                x=[c.upper() for c in channel_counts.keys()],
                y=list(channel_counts.values()),
                marker_color=NAVY,
            ))
            _styled_fig(ch_fig)
            ch_fig.update_layout(title="Actions by Channel")
        else:
            ch_fig = _empty_fig("Actions by Channel — waiting...")

        # Action by measure (with full names in hover)
        if actions:
            measure_counts = Counter(a.get("measure", "none") for a in actions if a.get("action_id", 0) != 0)
            top_measures = measure_counts.most_common(10)
            labels = [m for m, _ in top_measures]
            values = [c for _, c in top_measures]
            hover = [f"{m} — {MEASURE_DESCRIPTIONS.get(m, m)}<br>Count: {c}" for m, c in top_measures]
            m_fig = go.Figure(go.Bar(x=labels, y=values, hovertext=hover, hoverinfo="text",
                                     marker_color=HUMANA_GREEN))
            _styled_fig(m_fig)
            m_fig.update_layout(title="Top Measures Targeted")
        else:
            m_fig = _empty_fig("Measures Targeted — waiting...")

        # Action vs no-action
        if actions:
            action_count = sum(1 for a in actions if a.get("action_id", 0) != 0)
            noaction_count = sum(1 for a in actions if a.get("action_id", 0) == 0)
            pie = go.Figure(go.Pie(
                labels=["Action", "No Action"], values=[action_count, noaction_count],
                marker=dict(colors=[HUMANA_GREEN, "#e2e8f0"]),
            ))
            _styled_fig(pie)
            pie.update_layout(title="Action vs No-Action")
        else:
            pie = _empty_fig("Action Distribution — waiting...")

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
                        m_full = MEASURE_DESCRIPTIONS.get(act.measure, act.measure)
                        short_label = f"{act.measure} | {act.channel.upper()} | {act.variant.replace('_', ' ').title()}"
                        hover = f"{act.measure} — {m_full}<br>Channel: {act.channel.upper()}<br>Variant: {act.variant.replace('_', ' ').title()}<br>Q-Value: {float(avg_q[aid]):.4f}"
                        action_data.append((short_label, float(avg_q[aid]), hover))

                action_data.sort(key=lambda x: x[1], reverse=True)
                top = action_data[:20]

                if top:
                    labels = [t[0] for t in top]
                    values = [t[1] for t in top]
                    hovers = [t[2] for t in top]
                    colors = [HUMANA_GREEN if v > 0 else "#94a3b8" for v in values]

                    leaderboard = go.Figure(go.Bar(
                        y=labels, x=values, orientation="h",
                        marker_color=colors,
                        text=[f"{v:.3f}" for v in values],
                        textposition="outside",
                        hovertext=hovers, hoverinfo="text",
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
            action_perf = defaultdict(lambda: {"total": 0, "accepted": 0, "completed": 0, "measure": "?"})
            for r in sm_data:
                measure = r.get("measure", "?")
                m_full = MEASURE_DESCRIPTIONS.get(measure, measure)
                channel = r.get("channel", "?").upper()
                variant = r.get("variant", "?").replace("_", " ").title()
                label = f"{measure} | {channel} | {variant}"
                state = r.get("current_state", "")
                action_perf[label]["total"] += 1
                action_perf[label]["measure"] = measure
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
                    f"{v['measure']} — {MEASURE_DESCRIPTIONS.get(v['measure'], v['measure'])}<br>"
                    f"{k}<br>"
                    f"Accepted: {v['accepted']}/{v['total']} ({v['accepted']/max(v['total'],1):.0%})"
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

        # --- Action Variant Breakdown Table ---
        sm_data_for_breakdown = load_all_state_machine_data()
        if sm_data_for_breakdown:
            from collections import defaultdict
            variant_stats = defaultdict(lambda: {"total": 0, "accepted": 0, "completed": 0})
            for r in sm_data_for_breakdown:
                measure = r.get("measure", "?")
                channel = r.get("channel", "?")
                variant = r.get("variant", "?")
                label = f"{measure} — {MEASURE_DESCRIPTIONS.get(measure, measure)}"
                ch_label = channel.upper()
                v_label = (variant or "").replace("_", " ").title()
                key = (label, ch_label, v_label)
                state = r.get("current_state", "")
                variant_stats[key]["total"] += 1
                if state in ("ACCEPTED", "COMPLETED"):
                    variant_stats[key]["accepted"] += 1
                if state == "COMPLETED":
                    variant_stats[key]["completed"] += 1

            sorted_variants = sorted(variant_stats.items(), key=lambda x: x[1]["total"], reverse=True)
            rows = []
            for (measure_label, ch, v), stats in sorted_variants[:30]:
                accept_rate = stats["accepted"] / max(stats["total"], 1)
                rows.append(html.Tr([
                    html.Td(measure_label, style={"fontSize": "11px"}),
                    html.Td(ch, style={"fontWeight": "600"}),
                    html.Td(v, style={"fontSize": "11px", "color": GRAY_TEXT}),
                    html.Td(f"{stats['total']:,}"),
                    html.Td(f"{accept_rate:.0%}", style={
                        "color": HUMANA_GREEN if accept_rate > 0.10 else ("#f59e0b" if accept_rate > 0.03 else "#ef4444"),
                        "fontWeight": "600"}),
                    html.Td(f"{stats['completed']:,}"),
                ]))
            variant_breakdown = html.Table([
                html.Thead(html.Tr([
                    html.Th("Measure"), html.Th("Channel"), html.Th("Variant"),
                    html.Th("Sent"), html.Th("Accept %"), html.Th("Completed"),
                ])),
                html.Tbody(rows),
            ], style={"width": "100%", "fontSize": "12px"})
        else:
            variant_breakdown = html.P("Waiting for action data...", style={"color": GRAY_TEXT})

        return budget_gauge, leaderboard, table, ch_fig, m_fig, pie, variant_breakdown

    # =========================================================================
    # Tab 3: Training & Simulation
    # =========================================================================
    @app.callback(
        [Output("champion-challenger", "figure"),
         Output("model-version-timeline", "figure"),
         Output("sim-performance-summary", "children"),
         Output("sim-action-distribution", "figure"),
         Output("sim-closure-predictions", "figure"),
         Output("sim-channel-effectiveness", "figure"),
         Output("sim-stars-projection", "figure"),
         Output("promotion-history-table", "children"),
         Output("sim-action-breakdown", "children"),
         Output("debug-losses", "figure"),
         Output("debug-q-values", "figure"),
         Output("debug-entropy-alpha", "figure"),
         Output("debug-cql-penalty", "figure")],
        Input("interval-component", "n_intervals"),
    )
    def update_training(_):
        metrics = load_cumulative_metrics()
        from dashboard.data_feed import load_sim_predictions
        sim_preds = load_sim_predictions()

        # --- Champion vs Challenger ---
        if metrics:
            days = [m["day"] for m in metrics if m.get("champion_score") is not None]
            champ = [m["champion_score"] for m in metrics if m.get("champion_score") is not None]
            chall = [m["challenger_score"] for m in metrics if m.get("challenger_score") is not None]
            cc_fig = go.Figure()
            cc_fig.add_trace(go.Scatter(x=days, y=champ, mode="lines+markers", name="Champion",
                                        line=dict(color=HUMANA_GREEN, width=2)))
            cc_fig.add_trace(go.Scatter(x=days, y=chall, mode="lines+markers", name="Challenger",
                                        line=dict(color=NAVY, width=2, dash="dot")))
            _styled_fig(cc_fig)
            cc_fig.update_layout(title="Champion vs Challenger (Learned World Reward)",
                               xaxis_title="Day", yaxis_title="Mean Reward")
        else:
            cc_fig = _empty_fig("Champion vs Challenger — waiting...")

        # --- Model version timeline ---
        if metrics:
            days = [m["day"] for m in metrics]
            versions = [m.get("model_version", 1) for m in metrics]
            mv_fig = go.Figure(go.Scatter(x=days, y=versions, mode="lines+markers",
                                          line=dict(color=HUMANA_GREEN, width=2),
                                          fill="tozeroy", fillcolor="rgba(0,166,100,0.1)"))
            _styled_fig(mv_fig)
            mv_fig.update_layout(title="Model Version Deployed", xaxis_title="Day", yaxis_title="Version")
        else:
            mv_fig = _empty_fig("Model versions — waiting...")

        # --- Simulation performance summary ---
        if sim_preds:
            latest = sim_preds[-1]
            no_act_rate = latest.get("no_action_rate", 0)
            mean_r = latest.get("mean_reward", 0)
            total_acts = latest.get("total_actions", 0)
            sim_summary = html.Div([
                html.Div([
                    html.Div([
                        html.Span(f"{mean_r:.3f}", style={"fontSize": "24px", "fontWeight": "700", "color": NAVY}),
                        html.Span(" predicted reward/episode", style={"fontSize": "13px", "color": GRAY_TEXT}),
                    ], style={"marginRight": "40px"}),
                    html.Div([
                        html.Span(f"{no_act_rate:.0%}", style={"fontSize": "24px", "fontWeight": "700", "color": NAVY}),
                        html.Span(" strategic silence rate", style={"fontSize": "13px", "color": GRAY_TEXT}),
                    ], style={"marginRight": "40px"}),
                    html.Div([
                        html.Span(f"{total_acts:,}", style={"fontSize": "24px", "fontWeight": "700", "color": NAVY}),
                        html.Span(f" simulated actions ({latest.get('n_episodes', 0)} episodes)", style={"fontSize": "13px", "color": GRAY_TEXT}),
                    ]),
                ], style={"display": "flex", "gap": "20px"}),
            ])
        else:
            sim_summary = html.P("Simulation predictions will appear after first nightly training...",
                               style={"color": GRAY_TEXT})

        # --- Simulated action distribution by measure ---
        if sim_preds:
            latest = sim_preds[-1]
            dist = latest.get("action_dist_by_measure", {})
            if dist:
                total_actions = sum(dist.values())
                measures = sorted(dist.keys(), key=lambda m: dist[m], reverse=True)
                pcts = [dist[m] / max(total_actions, 1) for m in measures]
                hover = [f"{m} — {MEASURE_DESCRIPTIONS.get(m, m)}<br>{dist[m]:,} actions ({dist[m]/max(total_actions,1):.1%})" for m in measures]
                act_dist = go.Figure(go.Bar(x=measures, y=pcts, marker_color=HUMANA_GREEN,
                                            hovertext=hover, hoverinfo="text",
                                            text=[f"{p:.0%}" for p in pcts], textposition="outside"))
                _styled_fig(act_dist)
                act_dist.update_layout(title="Predicted Action Mix by Measure (%)", yaxis_title="Share of Actions",
                                      yaxis=dict(tickformat=".0%"))
            else:
                act_dist = _empty_fig("No action distribution data yet")
        else:
            act_dist = _empty_fig("Waiting for simulation predictions...")

        # --- Simulated closure rates by measure ---
        if sim_preds:
            latest = sim_preds[-1]
            closure_rates = latest.get("sim_closure_rates", {})
            if closure_rates:
                measures = sorted(closure_rates.keys())
                rates = [closure_rates[m] for m in measures]
                hover = [f"{m} — {MEASURE_DESCRIPTIONS.get(m, m)}<br>Rate: {r:.1%}" for m, r in zip(measures, rates)]
                closure_fig = go.Figure(go.Bar(
                    x=measures, y=rates,
                    marker_color=[HUMANA_GREEN if r > 0.05 else "#94a3b8" for r in rates],
                    hovertext=hover, hoverinfo="text",
                    text=[f"{r:.1%}" for r in rates], textposition="outside",
                ))
                _styled_fig(closure_fig)
                closure_fig.update_layout(title="Predicted Closure Rate by Measure",
                                        yaxis_title="Closure Rate", yaxis=dict(tickformat=".0%"))
            else:
                closure_fig = _empty_fig("No closure predictions yet")
        else:
            closure_fig = _empty_fig("Waiting for simulation predictions...")

        # --- Simulated channel effectiveness ---
        if sim_preds:
            latest = sim_preds[-1]
            ch_rates = latest.get("sim_channel_rates", {})
            if ch_rates:
                channels = sorted(ch_rates.keys())
                rates = [ch_rates[c] for c in channels]
                ch_fig = go.Figure(go.Bar(
                    x=[c.upper() for c in channels], y=rates,
                    marker_color=NAVY,
                    text=[f"{r:.1%}" for r in rates], textposition="outside",
                ))
                _styled_fig(ch_fig)
                ch_fig.update_layout(title="Predicted Channel Effectiveness (Learned World)",
                                   yaxis_title="Closure Rate per Action")
            else:
                ch_fig = _empty_fig("No channel data yet")
        else:
            ch_fig = _empty_fig("Waiting for simulation predictions...")

        # --- Simulated STARS projection ---
        # Show the latest 90-day trajectory from the most recent nightly simulation
        if sim_preds and len(sim_preds) > 0:
            latest_pred = sim_preds[-1]
            trajectory = latest_pred.get("stars_trajectory", [])

            stars_proj = go.Figure()

            if trajectory:
                traj_days = [t["day"] for t in trajectory]
                traj_stars = [t["stars"] for t in trajectory]
                stars_proj.add_trace(go.Scatter(
                    x=traj_days, y=traj_stars, mode="lines+markers",
                    name="Simulated STARS", line=dict(color=HUMANA_GREEN, width=3),
                    marker=dict(size=6),
                ))

            # Also show final STARS from each past nightly run
            all_finals = [(p["day"], p.get("final_stars", 1.0)) for p in sim_preds]
            if len(all_finals) > 1:
                stars_proj.add_trace(go.Scatter(
                    x=[f[0] for f in all_finals], y=[f[1] for f in all_finals],
                    mode="markers", name="Nightly Final STARS",
                    marker=dict(size=10, color=NAVY, symbol="diamond"),
                ))

            stars_proj.add_hline(y=4.0, line_dash="dash", line_color="#ef4444",
                               annotation_text="4★ Bonus")
            _styled_fig(stars_proj)
            stars_proj.update_layout(
                title="Simulated 90-Day Quarter (Ground Truth World)",
                xaxis_title="Simulation Day", yaxis_title="STARS Score",
                yaxis=dict(range=[1, 5]),
            )
        else:
            stars_proj = _empty_fig("STARS projection — waiting for data...")

        # --- Promotion history ---
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
            table = html.Table([
                html.Thead(html.Tr([
                    html.Th("Day"), html.Th("Version"), html.Th("Old Score"), html.Th("New Score"),
                ])),
                html.Tbody(rows if rows else [html.Tr([html.Td("No promotions yet", colSpan=4)])]),
            ], style={"width": "100%"})
        else:
            table = html.P("Waiting for training data...")

        # --- Simulated action deployment breakdown ---
        if sim_preds:
            latest = sim_preds[-1]
            # Build from the detailed eval data stored in sim_predictions
            # We have action_dist_by_measure and action_dist_by_channel
            dist_m = latest.get("action_dist_by_measure", {})
            dist_c = latest.get("action_dist_by_channel", {})
            if dist_m:
                total_m = sum(dist_m.values())
                total_c = sum(dist_c.values()) if dist_c else total_m
                rows = []
                for m in sorted(dist_m.keys(), key=lambda x: dist_m[x], reverse=True):
                    m_full = MEASURE_DESCRIPTIONS.get(m, m)
                    pct = dist_m[m] / max(total_m, 1)
                    rows.append(html.Tr([
                        html.Td(f"{m} — {m_full}", style={"fontSize": "11px"}),
                        html.Td(f"{pct:.1%}"),
                        html.Td(f"{dist_m[m]:,}", style={"color": GRAY_TEXT, "fontSize": "11px"}),
                    ]))
                if dist_c:
                    rows.append(html.Tr([html.Td("", colSpan=3, style={"borderTop": f"2px solid {GRAY_BORDER}"})]))
                    for ch in sorted(dist_c.keys(), key=lambda x: dist_c[x], reverse=True):
                        pct = dist_c[ch] / max(total_c, 1)
                        rows.append(html.Tr([
                            html.Td(f"Channel: {ch.upper()}", style={"fontSize": "11px", "fontWeight": "600"}),
                            html.Td(f"{pct:.1%}"),
                            html.Td(f"{dist_c[ch]:,}", style={"color": GRAY_TEXT, "fontSize": "11px"}),
                        ]))
                sim_breakdown = html.Table([
                    html.Thead(html.Tr([html.Th("Action / Channel"), html.Th("Share"), html.Th("Count")])),
                    html.Tbody(rows),
                ], style={"width": "100%", "fontSize": "12px"})
            else:
                sim_breakdown = html.P("No predicted breakdown yet", style={"color": GRAY_TEXT})
        else:
            sim_breakdown = html.P("Waiting for simulation predictions...", style={"color": GRAY_TEXT})

        # --- CQL Training Debug Charts ---
        from dashboard.data_feed import load_training_debug
        debug_data = load_training_debug()

        if debug_data and len(debug_data) > 1:
            debug_days = [d["day"] for d in debug_data]

            # Losses over time
            losses_fig = go.Figure()
            losses_fig.add_trace(go.Scatter(
                x=debug_days, y=[d.get("final_critic_loss", 0) for d in debug_data],
                mode="lines+markers", name="Critic Loss", line=dict(color="#ef4444", width=2)))
            losses_fig.add_trace(go.Scatter(
                x=debug_days, y=[d.get("final_td_loss", 0) for d in debug_data],
                mode="lines+markers", name="TD Loss", line=dict(color="#f59e0b", width=2)))
            losses_fig.add_trace(go.Scatter(
                x=debug_days, y=[d.get("final_actor_loss", 0) for d in debug_data],
                mode="lines+markers", name="Actor Loss", line=dict(color=NAVY, width=2)))
            _styled_fig(losses_fig)
            losses_fig.update_layout(title="Training Losses (should converge)", xaxis_title="Day")

            # Q-values over time
            q_fig = go.Figure()
            q_means = [d.get("q_mean", 0) for d in debug_data]
            q_stds = [d.get("q_std", 0) for d in debug_data]
            q_fig.add_trace(go.Scatter(
                x=debug_days, y=q_means, mode="lines+markers", name="Q Mean",
                line=dict(color=HUMANA_GREEN, width=2)))
            q_fig.add_trace(go.Scatter(
                x=debug_days, y=[d.get("q_max", 0) for d in debug_data],
                mode="lines", name="Q Max", line=dict(color="#94a3b8", dash="dot")))
            q_fig.add_trace(go.Scatter(
                x=debug_days, y=[d.get("q_min", 0) for d in debug_data],
                mode="lines", name="Q Min", line=dict(color="#94a3b8", dash="dot")))
            _styled_fig(q_fig)
            q_fig.update_layout(title="Q-Value Distribution (watch for explosion/collapse)", xaxis_title="Day")

            # Entropy + Alpha
            ent_fig = go.Figure()
            ent_fig.add_trace(go.Scatter(
                x=debug_days, y=[d.get("final_entropy", 0) for d in debug_data],
                mode="lines+markers", name="Policy Entropy", line=dict(color=HUMANA_GREEN, width=2)))
            ent_fig.add_trace(go.Scatter(
                x=debug_days, y=[d.get("final_alpha", 0) for d in debug_data],
                mode="lines+markers", name="Alpha (temp)", line=dict(color=NAVY, width=2), yaxis="y2"))
            _styled_fig(ent_fig)
            ent_fig.update_layout(
                title="Policy Entropy & Temperature (entropy collapse = bad)",
                xaxis_title="Day",
                yaxis=dict(title="Entropy"),
                yaxis2=dict(title="Alpha", side="right", overlaying="y"),
            )

            # CQL penalty
            cql_fig = go.Figure()
            cql_fig.add_trace(go.Scatter(
                x=debug_days, y=[d.get("final_cql_penalty", 0) for d in debug_data],
                mode="lines+markers", name="CQL Penalty", line=dict(color="#ef4444", width=2)))
            _styled_fig(cql_fig)
            cql_fig.update_layout(title="CQL Conservative Penalty (too high = underestimates, too low = overestimates)",
                                xaxis_title="Day")
        else:
            losses_fig = _empty_fig("Training losses — waiting for data...")
            q_fig = _empty_fig("Q-values — waiting for data...")
            ent_fig = _empty_fig("Entropy — waiting for data...")
            cql_fig = _empty_fig("CQL penalty — waiting for data...")

        return cc_fig, mv_fig, sim_summary, act_dist, closure_fig, ch_fig, stars_proj, table, sim_breakdown, losses_fig, q_fig, ent_fig, cql_fig

    # =========================================================================
    # Tab 3b: Per-Night Training Curve (drill-down)
    # =========================================================================
    @app.callback(
        Output("debug-day-selector", "options"),
        Input("interval-component", "n_intervals"),
    )
    def update_debug_day_options(_):
        from dashboard.data_feed import load_training_debug
        debug_data = load_training_debug()
        if not debug_data:
            return []
        return [{"label": f"Day {d['day']}", "value": d["day"]} for d in debug_data]

    @app.callback(
        Output("debug-epoch-curve", "figure"),
        [Input("debug-day-selector", "value"),
         Input("interval-component", "n_intervals")],
    )
    def update_debug_epoch_curve(selected_day, _):
        from dashboard.data_feed import load_training_debug
        if not selected_day:
            return _empty_fig("Select a training day above...")

        debug_data = load_training_debug()
        day_data = next((d for d in debug_data if d.get("day") == selected_day), None)
        if not day_data:
            return _empty_fig(f"No data for day {selected_day}")

        # Prefer step_history (per-batch), fall back to loss_history (per-epoch)
        steps = day_data.get("step_history", [])
        if steps:
            x = [s["step"] for s in steps]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=[s["critic"] for s in steps],
                                     mode="lines", name="Critic", line=dict(color="#ef4444", width=1.5)))
            fig.add_trace(go.Scatter(x=x, y=[s["td"] for s in steps],
                                     mode="lines", name="TD", line=dict(color="#f59e0b", width=1.5)))
            fig.add_trace(go.Scatter(x=x, y=[s["actor"] for s in steps],
                                     mode="lines", name="Actor", line=dict(color=NAVY, width=1.5)))
            fig.add_trace(go.Scatter(x=x, y=[s["cql"] for s in steps],
                                     mode="lines", name="CQL Penalty", line=dict(color="#8b5cf6", width=1.5)))
            _styled_fig(fig)
            fig.update_layout(title=f"Day {selected_day} — Per-Step Loss Curve",
                            xaxis_title="Gradient Step", yaxis_title="Loss")
        else:
            history = day_data.get("loss_history", [])
            if history:
                epochs = [h["epoch"] for h in history]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=epochs, y=[h["critic"] for h in history],
                                         mode="lines+markers", name="Critic", line=dict(color="#ef4444")))
                fig.add_trace(go.Scatter(x=epochs, y=[h["td"] for h in history],
                                         mode="lines+markers", name="TD", line=dict(color="#f59e0b")))
                fig.add_trace(go.Scatter(x=epochs, y=[h["actor"] for h in history],
                                         mode="lines+markers", name="Actor", line=dict(color=NAVY)))
                _styled_fig(fig)
                fig.update_layout(title=f"Day {selected_day} — Per-Epoch Loss",
                                xaxis_title="Epoch", yaxis_title="Loss")
            else:
                fig = _empty_fig(f"No loss history for day {selected_day}")

        return fig

    # =========================================================================
    # Tab 4: Measure Deep Dive
    # =========================================================================
    @app.callback(
        [Output("measure-overview-card", "children"),
         Output("measure-closure-trend", "figure"),
         Output("measure-action-variants", "figure"),
         Output("measure-channel-effectiveness", "figure"),
         Output("measure-action-table", "children")],
        [Input("interval-component", "n_intervals"),
         Input("measure-selector", "value")],
    )
    def update_measures(_, selected_measure):
        from config import MEASURE_CUT_POINTS, CHANNELS
        from collections import defaultdict

        if not selected_measure:
            selected_measure = "COL"

        metrics = load_cumulative_metrics()
        sm_data = load_all_state_machine_data()
        measure_sm = [r for r in sm_data if r.get("measure") == selected_measure]
        measure_desc = MEASURE_DESCRIPTIONS.get(selected_measure, selected_measure)
        cuts = MEASURE_CUT_POINTS.get(selected_measure, {})
        cut_4star = cuts.get(4, 0.70)
        weight = MEASURE_WEIGHTS.get(selected_measure, 1)

        # --- Overview card ---
        current_rate = 0.0
        current_stars = 1.0
        if metrics:
            latest = metrics[-1]
            detail = latest.get("measure_detail", {}).get(selected_measure, {})
            current_rate = detail.get("rate", 0)
            current_stars = detail.get("stars", 1.0)

        total_actions_for_measure = len(measure_sm)
        unique_patients = len(set(r.get("patient_id") for r in measure_sm)) if measure_sm else 0

        overview = html.Div([
            html.Div([
                html.Div([
                    html.H3(f"{selected_measure} — {measure_desc}", style={
                        "margin": "0", "fontSize": "18px", "fontWeight": "700", "color": NAVY}),
                    html.Span(f"Weight: {weight}x", style={
                        "backgroundColor": NAVY if weight > 1 else "#e2e8f0",
                        "color": "white" if weight > 1 else GRAY_TEXT,
                        "padding": "2px 10px", "borderRadius": "12px",
                        "fontSize": "11px", "fontWeight": "600", "marginLeft": "12px"}),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
                html.Div([
                    html.Div([
                        html.Span(f"{current_rate:.1%}", style={"fontSize": "28px", "fontWeight": "700", "color": NAVY}),
                        html.Span(" closure rate", style={"fontSize": "13px", "color": GRAY_TEXT}),
                    ], style={"marginRight": "40px"}),
                    html.Div([
                        html.Span(f"{'★' * int(current_stars)}{'☆' * (5 - int(current_stars))} {current_stars:.1f}",
                                 style={"fontSize": "20px", "fontWeight": "600",
                                        "color": HUMANA_GREEN if current_stars >= 4 else ("#f59e0b" if current_stars >= 3 else "#ef4444")}),
                        html.Span(f" (need {cut_4star:.0%} for 4★)", style={"fontSize": "12px", "color": GRAY_TEXT}),
                    ], style={"marginRight": "40px"}),
                    html.Div([
                        html.Span(f"{total_actions_for_measure:,}", style={"fontSize": "28px", "fontWeight": "700", "color": NAVY}),
                        html.Span(" actions sent", style={"fontSize": "13px", "color": GRAY_TEXT}),
                    ], style={"marginRight": "40px"}),
                    html.Div([
                        html.Span(f"{unique_patients:,}", style={"fontSize": "28px", "fontWeight": "700", "color": NAVY}),
                        html.Span(" patients contacted", style={"fontSize": "13px", "color": GRAY_TEXT}),
                    ]),
                ], style={"display": "flex", "flexWrap": "wrap", "gap": "10px"}),
            ]),
        ], style={
            "background": "white", "borderRadius": "12px", "padding": "20px",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.08)", "border": f"1px solid {GRAY_BORDER}",
        })

        # --- Closure trend ---
        cut_5star = cuts.get(5, 0.85)
        if metrics:
            days = [m["day"] for m in metrics]
            rates = [m.get("measure_closure_rates", {}).get(selected_measure, 0) for m in metrics]
            trend = go.Figure()
            trend.add_trace(go.Scatter(x=days, y=rates, mode="lines+markers", name="Closure Rate",
                                       line=dict(color=HUMANA_GREEN, width=3), marker=dict(size=6)))
            trend.add_hline(y=cut_4star, line_dash="dash", line_color="#f59e0b",
                          annotation_text=f"4★ ({cut_4star:.0%})")
            trend.add_hline(y=cut_5star, line_dash="dot", line_color=HUMANA_GREEN,
                          annotation_text=f"5★ ({cut_5star:.0%})")
            _styled_fig(trend)
            trend.update_layout(title=f"Closure Rate Over Time", xaxis_title="Day",
                              yaxis_title="Rate", yaxis=dict(range=[0, 1]))
        else:
            trend = _empty_fig("Waiting for data...")

        # --- Action variant performance ---
        if measure_sm:
            variant_stats = defaultdict(lambda: {"total": 0, "accepted": 0, "completed": 0})
            for r in measure_sm:
                v = r.get("variant", "unknown")
                ch = r.get("channel", "?")
                key = f"{ch.upper()} — {v.replace('_', ' ').title()}"
                state = r.get("current_state", "")
                variant_stats[key]["total"] += 1
                if state in ("ACCEPTED", "COMPLETED"):
                    variant_stats[key]["accepted"] += 1
                if state == "COMPLETED":
                    variant_stats[key]["completed"] += 1

            # Sort by acceptance rate
            sorted_variants = sorted(variant_stats.items(),
                                    key=lambda x: x[1]["accepted"] / max(x[1]["total"], 1),
                                    reverse=True)
            labels = [k for k, _ in sorted_variants]
            rates = [v["accepted"] / max(v["total"], 1) for _, v in sorted_variants]
            hover = [f"{k}<br>Accepted: {v['accepted']}/{v['total']} ({v['accepted']/max(v['total'],1):.0%})<br>"
                    f"Completed: {v['completed']}/{v['total']}"
                    for k, v in sorted_variants]
            colors = [HUMANA_GREEN if r > 0.10 else (NAVY if r > 0.03 else "#94a3b8") for r in rates]

            variants_fig = go.Figure(go.Bar(
                y=labels, x=rates, orientation="h",
                marker_color=colors, text=[f"{r:.0%}" for r in rates],
                textposition="outside", hovertext=hover, hoverinfo="text",
            ))
            _styled_fig(variants_fig)
            variants_fig.update_layout(title="Action Variants Ranked by Acceptance",
                                      xaxis_title="Acceptance Rate",
                                      yaxis=dict(autorange="reversed"))
        else:
            variants_fig = _empty_fig(f"No actions for {selected_measure} yet")

        # --- Channel effectiveness ---
        if measure_sm:
            ch_stats = defaultdict(lambda: {"total": 0, "accepted": 0, "completed": 0})
            for r in measure_sm:
                ch = r.get("channel", "unknown")
                state = r.get("current_state", "")
                ch_stats[ch]["total"] += 1
                if state in ("ACCEPTED", "COMPLETED"):
                    ch_stats[ch]["accepted"] += 1
                if state == "COMPLETED":
                    ch_stats[ch]["completed"] += 1

            channels = sorted(ch_stats.keys())
            accept_rates = [ch_stats[c]["accepted"] / max(ch_stats[c]["total"], 1) for c in channels]
            volumes = [ch_stats[c]["total"] for c in channels]
            hover = [f"{c.upper()}<br>Volume: {ch_stats[c]['total']}<br>"
                    f"Accepted: {ch_stats[c]['accepted']} ({ch_stats[c]['accepted']/max(ch_stats[c]['total'],1):.0%})<br>"
                    f"Completed: {ch_stats[c]['completed']}"
                    for c in channels]
            eff = go.Figure(go.Bar(
                x=[c.upper() for c in channels], y=accept_rates,
                marker_color=HUMANA_GREEN, hovertext=hover, hoverinfo="text",
            ))
            _styled_fig(eff)
            eff.update_layout(title="Channel Acceptance Rate", yaxis_title="Rate")
        else:
            eff = _empty_fig(f"No channel data for {selected_measure}")

        # --- Action deployment table ---
        if measure_sm:
            action_stats = defaultdict(lambda: {"total": 0, "presented": 0, "viewed": 0,
                                                "accepted": 0, "completed": 0, "failed": 0, "expired": 0})
            for r in measure_sm:
                ch = r.get("channel", "?")
                v = r.get("variant", "?")
                key = f"{ch.upper()} | {v.replace('_', ' ').title()}"
                state = r.get("current_state", "")
                action_stats[key]["total"] += 1
                for s in ["PRESENTED", "VIEWED", "ACCEPTED", "COMPLETED", "FAILED", "EXPIRED"]:
                    if state == s:
                        action_stats[key][s.lower()] += 1

            sorted_actions = sorted(action_stats.items(), key=lambda x: x[1]["total"], reverse=True)
            rows = []
            for action_name, stats in sorted_actions:
                accept_rate = stats["accepted"] / max(stats["total"], 1)
                complete_rate = stats["completed"] / max(stats["total"], 1)
                # Flag for potential discontinuation
                flag = ""
                if stats["total"] > 10 and accept_rate < 0.02:
                    flag = "⚠️ Consider discontinuing"

                rows.append(html.Tr([
                    html.Td(action_name, style={"fontWeight": "500", "fontSize": "12px"}),
                    html.Td(f"{stats['total']:,}"),
                    html.Td(f"{accept_rate:.0%}", style={
                        "color": HUMANA_GREEN if accept_rate > 0.10 else ("#f59e0b" if accept_rate > 0.03 else "#ef4444"),
                        "fontWeight": "600"}),
                    html.Td(f"{complete_rate:.0%}"),
                    html.Td(f"{stats['failed'] + stats['expired']}"),
                    html.Td(flag, style={"color": "#f59e0b", "fontSize": "11px"}),
                ]))

            action_table = html.Table([
                html.Thead(html.Tr([
                    html.Th("Action"), html.Th("Sent"), html.Th("Accept %"),
                    html.Th("Complete %"), html.Th("Failed"), html.Th("Flag"),
                ])),
                html.Tbody(rows),
            ], style={"width": "100%", "fontSize": "12px"})
        else:
            action_table = html.P(f"No actions deployed for {selected_measure} yet",
                                style={"color": GRAY_TEXT})

        return overview, trend, variants_fig, eff, action_table

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
        empty_fig = _empty_fig("Select a patient above...")

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

        # --- Action Cards grouped by week ---
        ch_icons = {"sms": "📱", "email": "📧", "portal": "🌐", "app": "📲", "ivr": "📞"}
        ch_bg_colors = {"sms": "#e3f2fd", "email": "#fff3e0", "portal": "#e8f5e9",
                        "app": "#fce4ec", "ivr": "#f3e5f5"}

        # Group actions by week (skip no-actions)
        from collections import defaultdict
        weeks = defaultdict(list)
        for a in journey:
            if a.get("action_id", 0) == 0:
                continue
            day = a.get("day", 0)
            week_num = (day - 1) // 7 + 1
            weeks[week_num].append(a)

        # Find all weeks that have data
        if weeks:
            max_week = max(weeks.keys())
        else:
            max_week = 1

        cards = []
        for week in range(1, max_week + 1):
            week_actions = weeks.get(week, [])
            if not week_actions:
                # Collapsed placeholder for empty weeks
                cards.append(html.Div(
                    f"Week {week} — No actions",
                    style={"color": GRAY_TEXT, "fontSize": "12px", "padding": "6px 16px",
                           "borderLeft": f"3px solid {GRAY_BORDER}", "marginBottom": "4px"},
                ))
                continue

            # Week card with actions inside
            action_cards_in_week = []
            for a in week_actions:
                eng = a.get("engagement", {})
                day = a.get("day", "?")
                measure = a.get("measure", "?")
                measure_full = MEASURE_DESCRIPTIONS.get(measure, measure)
                channel = a.get("channel", "?")
                variant = (a.get("variant", "") or "").replace("_", " ").title()
                reward = a.get("reward", 0)

                # Disposition
                if eng.get("clicked") or eng.get("completed"):
                    disp_icon, disp_text, disp_color = "👍", "Clicked", "#4caf50"
                    border = f"2px solid #4caf50"
                elif eng.get("opened"):
                    disp_icon, disp_text, disp_color = "👁️", "Viewed", "#ff9800"
                    border = f"2px solid #ff9800"
                elif eng.get("delivered"):
                    disp_icon, disp_text, disp_color = "📬", "Delivered", "#2196f3"
                    border = f"1px solid #2196f3"
                elif eng.get("failed"):
                    disp_icon, disp_text, disp_color = "👎", "Failed", "#f44336"
                    border = f"2px solid #f44336"
                else:
                    disp_icon, disp_text, disp_color = "📤", "Sent", "#666"
                    border = f"1px solid #ddd"

                action_cards_in_week.append(html.Div([
                    html.Div([
                        html.Span(f"Day {day}", style={"fontWeight": "bold", "fontSize": "12px"}),
                        html.Span(f" {ch_icons.get(channel, '❓')} {channel.upper()}", style={
                            "fontSize": "11px", "marginLeft": "8px", "color": "#555"}),
                    ], style={"marginBottom": "4px"}),
                    html.Div([
                        html.Span(measure, style={
                            "backgroundColor": NAVY, "color": "white", "padding": "2px 8px",
                            "borderRadius": "12px", "fontSize": "10px", "fontWeight": "bold"}),
                        html.Span(f" {measure_full}", style={"fontSize": "10px", "color": GRAY_TEXT, "marginLeft": "4px"}),
                    ], style={"marginBottom": "4px"}),
                    html.Div(variant, style={"fontSize": "11px", "color": "#555", "marginBottom": "4px"}),
                    html.Div([
                        html.Span(f"{disp_icon} ", style={"fontSize": "16px"}),
                        html.Span(disp_text, style={"color": disp_color, "fontWeight": "bold", "fontSize": "12px"}),
                        html.Span(f" {reward:+.3f}", style={"fontSize": "11px", "color": "#888", "marginLeft": "auto"}),
                    ], style={"display": "flex", "alignItems": "center"}),
                ], style={
                    "border": border, "borderRadius": "8px", "padding": "8px 12px",
                    "backgroundColor": ch_bg_colors.get(channel, "#f5f5f5"),
                    "minWidth": "180px", "maxWidth": "220px",
                    "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
                }))

            cards.append(html.Div([
                html.Div(f"Week {week}", style={
                    "fontWeight": "600", "fontSize": "13px", "color": NAVY,
                    "marginBottom": "8px", "paddingBottom": "4px",
                    "borderBottom": f"2px solid {HUMANA_GREEN}",
                }),
                html.Div(action_cards_in_week, style={
                    "display": "flex", "flexWrap": "wrap", "gap": "10px",
                }),
            ], style={
                "background": "white", "borderRadius": "10px", "padding": "12px 16px",
                "border": f"1px solid {GRAY_BORDER}", "marginBottom": "10px",
            }))

        # --- Reward Curve ---
        rewards = [a.get("reward", 0) for a in journey]
        cum_rewards = []
        total = 0
        for r in rewards:
            total += r
            cum_rewards.append(total)
        reward_fig = go.Figure()
        reward_fig.add_trace(go.Scatter(y=cum_rewards, mode="lines+markers", name="Cumulative Reward",
                                        line=dict(color=HUMANA_GREEN, width=2)))
        _styled_fig(reward_fig)
        reward_fig.update_layout(title="Cumulative Reward", xaxis_title="Interaction", yaxis_title="Reward")

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
            empty = _empty_fig("Waiting for state machine data...")
            return empty, empty, html.P("No data yet"), empty

        # --- Funnel: cumulative "reached this stage or beyond" ---
        total = len(sm_data)
        from simulation.action_state_machine import LIFECYCLE_STAGES
        stage_order = [s.value for s in LIFECYCLE_STAGES]
        reached = {s: 0 for s in stage_order}
        for r in sm_data:
            states_visited = {sh.get("state", sh) for sh in r.get("state_history", [])}
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

        # --- Channel conversion rates ---
        channel_states: dict = {}
        for r in sm_data:
            ch = r.get("channel", "unknown")
            state = r.get("current_state", "UNKNOWN")
            if ch not in channel_states:
                channel_states[ch] = Counter()
            channel_states[ch][state] += 1

        if channel_states:
            ch_funnel = go.Figure()
            for ch, counts in channel_states.items():
                presented = sum(counts.get(s, 0) for s in ["PRESENTED", "VIEWED", "ACCEPTED", "COMPLETED", "DECLINED"])
                viewed = sum(counts.get(s, 0) for s in ["VIEWED", "ACCEPTED", "COMPLETED", "DECLINED"])
                accepted = sum(counts.get(s, 0) for s in ["ACCEPTED", "COMPLETED"])
                completed = counts.get("COMPLETED", 0)
                ch_total = sum(counts.values())
                ch_funnel.add_trace(go.Bar(
                    name=ch.upper(),
                    x=["Presented", "Viewed", "Accepted", "Completed"],
                    y=[presented / max(ch_total, 1), viewed / max(ch_total, 1),
                       accepted / max(ch_total, 1), completed / max(ch_total, 1)],
                ))
            _styled_fig(ch_funnel)
            ch_funnel.update_layout(title="Conversion by Channel", barmode="group", yaxis_title="Rate")
        else:
            ch_funnel = _empty_fig("Channel conversion — waiting for data...")

        # --- Recent transitions table ---
        recent = sm_data[-30:]
        rows = []
        for r in reversed(recent):
            current = r.get("current_state", "?")
            measure = r.get("measure", "")
            measure_full = f"{measure} — {MEASURE_DESCRIPTIONS.get(measure, '')}" if measure else ""
            rows.append(html.Tr([
                html.Td(r.get("tracking_id", "")[:25]),
                html.Td(r.get("patient_id", "")),
                html.Td(measure_full, style={"fontSize": "11px"}),
                html.Td(r.get("channel", "").upper()),
                html.Td(current, style={
                    "color": HUMANA_GREEN if current == "COMPLETED" else ("#ef4444" if current in ("FAILED", "EXPIRED") else NAVY)
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

        # --- Daily gap closure efficiency: closures per action over time ---
        # Shows whether the model is getting better at picking actions that close gaps
        metrics = load_cumulative_metrics()
        if metrics and len(metrics) > 1:
            conv = go.Figure()
            days_m = [m["day"] for m in metrics]
            # Closures per action (efficiency)
            closures_per_action = []
            daily_actions_list = []
            daily_closures_list = []
            for m in metrics:
                actions = max(m.get("daily_actions", 1), 1)
                closures = m.get("daily_closures", 0)
                closures_per_action.append(closures / actions)
                daily_actions_list.append(actions)
                daily_closures_list.append(closures)

            conv.add_trace(go.Scatter(x=days_m, y=closures_per_action, mode="lines+markers",
                                      name="Closures per Action", line=dict(color=HUMANA_GREEN, width=2)))
            conv.add_trace(go.Bar(x=days_m, y=daily_closures_list, name="Daily Closures",
                                  marker_color="rgba(0,166,100,0.2)", yaxis="y2"))
            _styled_fig(conv)
            conv.update_layout(
                title="Gap Closure Efficiency (closures per action sent)",
                xaxis_title="Day",
                yaxis=dict(title="Closures / Action", side="left"),
                yaxis2=dict(title="Daily Closures", side="right", overlaying="y"),
                legend=dict(x=0.01, y=0.99),
            )
        else:
            conv = _empty_fig("Closure efficiency — waiting for data...")

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
