# HEDIS STARS Gap Closure — RL Agent Framework

An offline reinforcement learning system that optimizes omnichannel patient outreach for Medicare Advantage STARS gap closure. The agent learns **which messages to send, through which channels, to which patients — and critically, when to stay silent**. It balances a dual objective: closing HEDIS care gaps to achieve a STARS rating above 4.0 for CMS bonus payments, while conserving a finite per-patient message budget to reduce fatigue and preserve outreach capacity for high-impact moments.

## Problem

Medicare Advantage health plans are rated on a 1–5 star scale by CMS based on quality measures (HEDIS). Plans scoring **≥ 4.0 stars** receive bonus payments worth millions in additional revenue. Closing care gaps — ensuring patients complete screenings, maintain medication adherence, manage chronic conditions — is the primary lever for improving STARS scores.

Today, outreach is rule-based: static campaign schedules push the same messages regardless of patient context. This project replaces that with a **Conservative Q-Learning (CQL) agent** that learns an optimal outreach policy from historical data, respects real-world constraints (opt-outs, contact limits, channel availability), and continuously improves through nightly retraining.

## Objective

Train and deploy an RL agent that:

1. **Selects the best action** (measure × channel × content variant) for each patient each day
2. **Learns when to do nothing** — each patient has a finite message budget (12/quarter, 45/year); the agent learns to conserve messages for high-impact moments rather than exhausting the budget on low-value contacts
3. **Respects eligibility constraints** — SMS consent, app install status, contact frequency limits, suppression rules, budget exhaustion
4. **Handles lagged rewards** — gap closures may occur weeks after outreach; a learned reward model bridges the delay
5. **Improves nightly** — each night a challenger model is trained on all accumulated data and promoted if it outperforms the current champion
6. **Targets STARS ≥ 4.0** — reward function weights triple-weighted measures (medication adherence) proportionally

## Quick Start

```bash
# Clone and install
git clone <repo-url> && cd rl_agent_foundations
pip install -e ".[dev]"

# Run everything (generates data, starts dashboard, runs 30-day simulation)
./run.sh

# Or step by step:
./run.sh generate    # Generate 5,000-patient mock dataset
./run.sh dashboard   # Start dashboard on http://localhost:8050
./run.sh simulate    # Run 30-day simulation

# Management
./run.sh status      # Check what's running + simulation progress
./run.sh logs        # Tail simulation log
./run.sh stop        # Stop all processes
./run.sh restart     # Stop + fresh start
./run.sh clean       # Wipe all data and stop
```

Open **http://localhost:8050** to watch the simulation in real-time across 7 dashboard tabs.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        30-Day Simulation Loop                       │
│                                                                     │
│   Day 0: Generate Data → BC Training → CQL Training → Deploy v1    │
│                                                                     │
│   Days 1–30:                                                        │
│   ┌──────────────────────┐     ┌──────────────────────────────┐    │
│   │      DAY PHASE       │     │        NIGHT PHASE           │    │
│   │                      │     │                              │    │
│   │  Agent selects       │     │  Aggregate all experience    │    │
│   │  actions for 5,000   │────▶│  Train challenger CQL model  │    │
│   │  patients             │     │  Evaluate champ vs chall     │    │
│   │  State machine tracks │     │  Promote if >2% improvement  │    │
│   │  action lifecycle     │     │  Log training metrics        │    │
│   │  Lagged rewards queue │     │                              │    │
│   └──────────────────────┘     └──────────────────────────────┘    │
│          │                                    │                     │
│          ▼                                    ▼                     │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │              data/simulation/day_XX/                      │     │
│   │   actions_taken.json · experience_buffer.json             │     │
│   │   state_machine.json · nightly_metrics.json               │     │
│   └──────────────────────────────────────────────────────────┘     │
│                              │                                      │
│                              ▼                                      │
│                    ┌───────────────────┐                            │
│                    │  Plotly Dash App  │                            │
│                    │  localhost:8050   │                            │
│                    │  7 tabs, 5s poll  │                            │
│                    └───────────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘
```

## HEDIS Measures (18 Part C Measures)

| Category | Measures | STARS Weight |
|----------|----------|:---:|
| Screenings | COL (Colorectal), BCS (Breast Cancer), EED (Eye Exam — Diabetes) | 1× |
| Vaccines | FVA (Tdap), FVO (Pneumococcal), AIS (Zoster), FLU (Influenza) | 1× |
| Chronic Management | CBP (Blood Pressure), BPD (BP — Diabetes), HBD (A1C Control), KED (Kidney) | 1× |
| Medication Adherence | MAC (Statins), MRA (RAS Antagonists), MDS (Diabetes Oral) | **3×** |
| Mental Health | DSF (Depression Screening), DRR (Depression Remission), DMC02 (Antidepressants) | 1× (DMC02 = 3×) |
| Care Coordination | TRC_M (Medication Reconciliation Post-Discharge) | 3× |

Triple-weighted measures (MAC, MRA, MDS, DMC02, TRC_M) have 3× impact on the STARS score and receive 3× the gap closure reward.

## Action Space (125 Curated Actions)

Each action is a realistic **(measure, channel, content_variant)** tuple — not a full combinatorial explosion. Only sensible combinations are included.

**5 Channels:** SMS, Email, Web Portal, Mobile App, IVR

**Example actions by category:**

| Category | Channel | Variant | Example Message |
|----------|---------|---------|-----------------|
| Screening | SMS | `scheduling_link` | "Book your colonoscopy — tap here to schedule" |
| Screening | SMS | `incentive_offer` | "$50 reward when you complete your screening" |
| Vaccine | SMS | `pharmacy_locator` | "Get your flu shot free at CVS near you" |
| Med Adherence | SMS | `refill_reminder` | "Time to refill your statin — tap to order" |
| Med Adherence | App | `adherence_gamification` | Streak tracker + reward milestones |
| Chronic Mgmt | SMS | `home_device_offer` | "Free BP monitor — reply YES to enroll" |
| Mental Health | SMS | `telehealth_link` | "Talk to a therapist today — free virtual visit" |
| Care Transition | IVR | `care_navigator` | Automated call with live transfer to transition nurse |

Plus **1 no-action** (index 0) — the agent can choose not to contact a patient.

Action index 0 is always valid. All other actions are subject to eligibility constraints (action masking).

## State Space (96 Features)

The patient state vector is constructed from clinical, behavioral, and temporal features:

| Block | Features | Dims |
|-------|----------|:---:|
| Demographics | age, sex, zip3, dual-eligible, LIS, SNP | 6 |
| Clinical Vitals | BP systolic/diastolic, A1C, BMI, CKD stage, PHQ-9 | 6 |
| Condition Flags | diabetes, hypertension, hyperlipidemia, depression, CKD, CHD, COPD, CHF | 8 |
| Medication Fill Rates | statin PDC, ACE/ARB PDC, diabetes oral PDC, antidepressant PDC | 4 |
| Open Gap Flags | one binary flag per HEDIS measure | 18 |
| Engagement | channel consent (4), response rates (5), contact count, days since contact | 11 |
| Risk Scores | readmission, disenrollment, non-compliance, composite acuity | 4 |
| Temporal | day-of-year (sin/cos), days remaining in measurement year | 3 |
| Action History | last 5 actions encoded as (measure_idx, channel_idx) | 10 |
| Gap-Specific | per open gap: days since last attempt, attempt count (top 5) | 10 |
| **Padding** | | **26** |
| **Total** | | **96** |

All features are normalized to roughly [0, 1]. The full vector is padded to 96 dimensions for tensor alignment.

## Reward Function

```
R(s, a) = gap_closure_reward × measure_weight     (1× or 3× for triple-weighted)
         + 0.05 × delivered + 0.10 × clicked       (immediate engagement signals)
         − 0.01                                     (action cost — prevents spam)
         − 0.05 × fatigue_factor                    (contact frequency penalty)
```

**Lagged rewards:** Gap closure is heavily delayed (weeks to months). A learned reward model predicts `P(gap_closure | state, action, days_elapsed)` to provide training signal before the actual outcome is observed. When the real gap closure arrives, it retroactively updates the training data.

## Model Architecture — Actor-Critic CQL

The agent uses **Conservative Q-Learning with a Soft Actor-Critic backbone**:

```
┌──────────────────────┐
│        Actor          │   Policy network: π(a|s)
│  96 → 256 → 256 → 125│   Outputs action probabilities with masking
│  + Action Masking     │   Invalid actions get P = 0
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐   ┌──────────────────────┐
│     Critic Q1         │   │     Critic Q2         │   Twin Q-Networks
│  96 → 256 → 256 → 125│   │  96 → 256 → 256 → 125│   Conservative estimate:
└──────────┬───────────┘   └──────────┬───────────┘   Q = min(Q1, Q2)
           │                          │
           └──────────┬───────────────┘
                      ▼
              CQL Penalty:
              LogSumExp(Q_valid) − Q(s, a_data)
              Pushes Q-values down for OOD actions
```

**Training pipeline:**
1. **Behavior Cloning** — initialize the actor from historical outreach data (what the business was doing)
2. **CQL Fine-Tuning** — conservative offline RL that improves on the behavioral policy without overestimating Q-values for unseen actions
3. **Nightly Retraining** — each simulated night, a challenger is trained on all accumulated data and promoted if it beats the champion by ≥ 2%

**Why CQL?** Standard Q-learning overestimates Q-values for out-of-distribution actions — dangerous in healthcare where we cannot experiment freely. CQL adds a penalty that keeps Q-values conservative, ensuring the learned policy stays close to the data distribution while still improving.

## Action Masking & Constraints

Not every action is valid for every patient. The action mask is a boolean vector of size 125 that enforces:

| Constraint | Effect |
|-----------|--------|
| Gap not open | All actions for that measure are blocked |
| No SMS consent | All SMS actions blocked |
| App not installed | All app actions blocked |
| Portal not registered | All portal actions blocked |
| Opt-out | Only no-action available |
| Grievance hold | Only no-action available |
| Suppression active | Only no-action available |
| Contact limit reached (3/week) | Only no-action available |
| Same-measure cooldown (7 days) | Actions for that measure blocked |

The mask is applied in both the actor (masked softmax) and critics (masked Q-values, CQL penalty only on valid actions).

## Message Budget & Strategic Silence

Each patient has a **finite message budget** that creates a fundamental tension: every message spent on a low-value contact is one fewer message available when it really matters.

```
┌─────────────────────────────────────────────────┐
│  Patient P10042 — Quarter 1 Budget              │
│                                                 │
│  ████████████░░░░  9/12 remaining (75%)         │
│  Status: NORMAL                                 │
│                                                 │
│  Quarter 2 resets to 12                         │
│  Annual cap: 45 messages                        │
└─────────────────────────────────────────────────┘
```

**Budget parameters:**
- **12 messages per quarter** (90-day rolling window with replenishment)
- **45 messages per year** (hard annual cap)
- **Warning at < 25%** remaining — agent penalized for messages that don't produce clicks
- **Critical at < 10%** remaining — heavy penalty for any non-essential message
- **Exhausted at 0** — all actions masked, only no_action available until next quarter

**How the agent learns strategic silence:**

1. **Budget in the state vector** — 4 features: `budget_remaining_norm`, `budget_utilization_rate`, `budget_is_warning`, `budget_is_critical`. The agent sees its remaining budget and learns to plan ahead.

2. **Budget-aware reward shaping:**
   - No-action gets a **positive reward** when budget is below 25% (conservation bonus of +0.02)
   - Messages that don't produce clicks get a **waste penalty** (-0.08) when budget is below 25%
   - Any message when budget is below 10% gets a **critical penalty** (-0.15) — only send if gap closure is very likely
   - Gap closure reward (+1.0 to +3.0) always dominates — but the agent learns that a well-timed message at 5% budget that closes a gap is worth far more than 5 wasted messages earlier

3. **Budget exhaustion as hard constraint** — when budget hits 0, action masking forces no_action. The agent cannot reach patients who need it. This creates a long-horizon planning problem: the agent must balance exploration (sending messages to learn what works) against exploitation (saving budget for patients most likely to respond).

This design naturally produces behavior where the agent:
- **Sends fewer, higher-quality messages** — targeting patients with high response probability
- **Prioritizes triple-weighted measures** (MAC, MRA, MDS) when budget is scarce
- **Avoids re-contacting** patients who haven't responded to prior outreach
- **Times messages** to when engagement signals suggest the patient is receptive
- **Preserves budget** for late-year urgency when approaching measurement year end

## Action Lifecycle State Machine

An external process tracks each action through its lifecycle:

```
CREATED → QUEUED → PRESENTED → VIEWED → ACCEPTED → COMPLETED
                            ↘ EXPIRED      ↘ DECLINED
                   ↘ FAILED
```

State transitions feed back into:
- **Engagement signals** → immediate reward (delivered, opened, clicked)
- **Eligibility updates** → pending actions block re-sends for the same measure
- **Dashboard** → action lifecycle funnel visualization

## Dashboard (7 Tabs)

| Tab | Content |
|-----|---------|
| **STARS Overview** | Gauge chart (target 4.0), score trajectory, cumulative reward curve, regret curve, per-measure closure table |
| **Real-Time Actions** | Live scrolling feed of actions taken, distribution by channel/measure, action vs no-action ratio |
| **Training Performance** | Champion vs challenger scores per day, model version timeline, promotion history |
| **Measure Deep Dive** | Select a HEDIS measure → closure trend, channel effectiveness, patient funnel (sent → delivered → viewed → clicked) |
| **Patient Journey** | Search by patient ID → full timeline of every action, engagement outcomes, cumulative reward curve, gap status |
| **Action Lifecycle** | State machine funnel, per-channel conversion rates, recent state transitions, terminal state distribution |
| **Logs** | Real-time simulation log stream with level filtering (PHASE, METRIC, INFO, ERROR) |

The dashboard polls `data/simulation/` JSON files every 5 seconds via `dcc.Interval`.

## Running Individual Modules

### Data Generation

```bash
# Generate all 4 mock datasets (5,000 patients)
python scripts/generate_data.py --cohort-size 5000 --seed 42
```

Produces:
- `data/generated/state_features.json` — patient state snapshots (demographics, clinical, engagement, risk)
- `data/generated/historical_activity.json` — 50,000 historical outreach records with outcomes
- `data/generated/gap_closure.json` — longitudinal gap closure timelines per patient per measure
- `data/generated/action_eligibility.json` — per-patient action masks with constraint reasons

### World Model Training

```bash
# Train dynamics model (predicts next state) and reward model (predicts gap closure probability)
python scripts/train_world_models.py --epochs 50 --batch-size 256
```

Saves to `training/checkpoints/dynamics_model.pt` and `reward_model.pt`.

### Agent Training

```bash
# Train BC → CQL pipeline
python scripts/train_agent.py --bc-epochs 50 --cql-epochs 100

# Skip BC if checkpoint exists
python scripts/train_agent.py --skip-bc --cql-epochs 100
```

Saves to `training/checkpoints/bc_policy.pt` and `cql_agent.pt`.

### Simulation

```bash
# Run 30-day simulation (includes BC + CQL training on Day 0)
python scripts/run_simulation.py --days 30 --bc-epochs 30 --cql-epochs 20 --eval-episodes 100

# Quick test run
python scripts/run_simulation.py --days 3 --bc-epochs 5 --cql-epochs 5 --eval-episodes 20
```

### Dashboard

```bash
# Start on default port 8050
python scripts/run_dashboard.py

# Custom port with debug mode
python scripts/run_dashboard.py --port 8080 --debug
```

### Tests

```bash
# Run full test suite (212 tests)
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_datagen.py -v        # Data generation + schema validation
python -m pytest tests/test_environment.py -v     # Env, masking, reward, gym
python -m pytest tests/test_training.py -v        # BC, CQL, evaluation
python -m pytest tests/test_simulation.py -v      # Daily/nightly cycles, state machine, metrics
python -m pytest tests/test_integration.py -v     # End-to-end, world models, edge cases
```

## Four Mock Datasets

| Dataset | Records | Purpose |
|---------|:---:|---------|
| **Historical Activity** | 50,000 | Past outreach actions + outcomes. Used for behavior cloning warm-start. Each record has action taken, delivery/engagement outcome, and lagged gap closure result. |
| **State Features** | 5,000 | Patient state snapshots. Demographics, clinical indicators (BP, A1C, medication fill rates), engagement history, risk scores. Flattened to 96-dim vectors. |
| **Gap Closure** | ~50,000 | Monthly timeline per patient per measure showing when gaps open/close. Used to train the reward model. |
| **Action Eligibility** | 5,000 | Per-patient action masks with constraint reasons. Encodes SMS consent, app install, opt-out, contact limits, and gap status. |

## Dependencies

```
gymnasium>=0.29.0
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.18.0
dash>=2.14.0
scipy>=1.11.0
scikit-learn>=1.3.0
```

## Deep Dive: Project Structure

```
rl_agent_foundations/
│
├── run.sh                           # One-command launcher (start/stop/restart/status/clean)
├── config.py                        # Central configuration
│                                    #   - 18 HEDIS measures with descriptions and weights
│                                    #   - 125 curated actions (Action namedtuple catalog)
│                                    #   - Lookup indices: ACTION_BY_ID, ACTION_IDS_BY_MEASURE
│                                    #   - State dim, cohort size, simulation days
│                                    #   - Reward weights, CQL hyperparams, lag distributions
│                                    #   - World model configs, dashboard settings, file paths
│
├── datagen/                         # Mock data generation
│   ├── constants.py                 #   Clinical ranges (BP, A1C, BMI, PHQ-9 distributions)
│   │                                #   Demographics (age, sex, ZIP3, condition prevalence)
│   │                                #   Channel rates (delivery, open, click per channel)
│   │                                #   Gap closure base rates, outreach lift factors
│   ├── patients.py                  #   Generate patient profiles with age-adjusted conditions
│   ├── state_features.py            #   Build clinical snapshots with condition-specific gaps
│   │                                #   Enforces: BCS female-only, diabetes measures need dx, etc.
│   ├── historical_activity.py       #   Simulate behavioral policy outreach with realistic
│   │                                #   engagement funnels and lagged gap closure outcomes
│   ├── gap_closure.py               #   Monthly timelines with urgency-adjusted closure rates
│   │                                #   and medication adherence gap reopening
│   ├── action_eligibility.py        #   Constraint-based masking: consent, install, opt-out,
│   │                                #   contact limits, gap status, per-action suppression
│   └── generator.py                 #   Orchestrates all generators, writes JSON to data/generated/
│
├── environment/                     # Gymnasium environment
│   ├── action_space.py              #   Encode/decode 125 actions ↔ (measure, channel, variant)
│   ├── state_space.py               #   Flatten patient dicts → 96-dim float32 vectors
│   │                                #   Normalized features, sin/cos temporal encoding
│   ├── action_masking.py            #   Compute boolean masks from gaps, channels, constraints
│   │                                #   Handles: opt-out, grievance, suppression, contact limit,
│   │                                #   channel unavailability, same-measure cooldown
│   ├── reward.py                    #   Composite reward: gap closure (weighted) + engagement
│   │                                #   + action cost + fatigue penalty
│   │                                #   compute_stars_score(): weighted avg → 1-5 star mapping
│   └── hedis_env.py                 #   Full gymnasium.Env with Dict observation space
│                                    #   Supports pluggable dynamics/reward models
│                                    #   Used for champion vs challenger evaluation
│
├── models/                          # PyTorch world models
│   ├── dynamics_model.py            #   s' = s + f_θ(s, embed(a)) + ε
│   │                                #   Embedding(125, 32) → MLP(128→256→256→96)
│   │                                #   Predicts state deltas with learned variance
│   ├── reward_model.py              #   P(closure | s, a, days) via sigmoid MLP
│   │                                #   Embedding(125, 32) → MLP(129→128→64→1)
│   │                                #   Binary cross-entropy on gap closure labels
│   ├── train_dynamics.py            #   Build (s, a, s') transitions from historical data
│   │                                #   Gaussian NLL loss, gradient clipping
│   └── train_reward.py              #   Build (s, a, days, closed) from activity + gap data
│                                    #   Multi-horizon samples (30d, 90d, actual closure day)
│
├── training/                        # RL training pipeline
│   ├── data_loader.py               #   JSON datasets → offline RL episodes
│   │                                #   Builds (obs, action, reward, mask) sequences per patient
│   │                                #   Also exports to RLlib JSON-lines format
│   ├── behavior_cloning.py          #   ActionMaskedPolicy: MLP with masked softmax
│   │                                #   Cross-entropy loss on historical actions
│   │                                #   Provides warm-start for CQL actor
│   ├── cql_trainer.py               #   Actor-Critic CQL (SAC backbone)
│   │                                #   Actor: masked softmax policy π(a|s)
│   │                                #   TwinCritic: Q1, Q2 with q_min = min(Q1, Q2)
│   │                                #   CQL loss: LogSumExp(Q_valid) − Q(s, a_data)
│   │                                #   Auto-tuned entropy coefficient α
│   │                                #   Soft target network updates (τ = 0.005)
│   └── evaluation.py               #   Run agent on env for N episodes
│                                    #   Compare champion vs challenger with improvement threshold
│
├── simulation/                      # 30-day simulation engine
│   ├── loop.py                      #   Main orchestrator: Day 0 init → daily/nightly loop
│   │                                #   Writes cumulative_metrics.json for dashboard
│   ├── daily_cycle.py               #   Agent selects actions for all patients
│   │                                #   Feeds state machine, schedules lagged rewards
│   │                                #   Writes actions_taken.json, experience_buffer.json
│   ├── nightly_cycle.py             #   Trains challenger CQL on historical + simulation data
│   │                                #   Evaluates both models, promotes if better
│   │                                #   Writes nightly_metrics.json
│   ├── lagged_rewards.py            #   LaggedRewardQueue: schedule closure at future day
│   │                                #   Measure-specific lag distributions (vaccines ~5d,
│   │                                #   screenings ~30d, medication adherence ~60d)
│   ├── action_state_machine.py      #   ActionLifecycleTracker: CREATED→QUEUED→PRESENTED→
│   │                                #   VIEWED→ACCEPTED/DECLINED→COMPLETED/FAILED/EXPIRED
│   │                                #   Channel-specific transition probabilities
│   │                                #   Feeds engagement signals + eligibility updates
│   ├── metrics.py                   #   MetricsTracker: STARS scores, cumulative reward,
│   │                                #   regret curves, per-measure closure rates
│   └── logger.py                    #   Structured JSONL logger for dashboard Logs tab
│
├── dashboard/                       # Plotly Dash real-time monitoring
│   ├── app.py                       #   Dash entry point, 7 tabs, 5s polling interval
│   ├── callbacks.py                 #   All callback functions for real-time chart updates
│   ├── data_feed.py                 #   Reads simulation JSONs, patient journey queries
│   └── layouts/
│       ├── overview.py              #   STARS gauge, trajectory, reward curve, measure table
│       ├── realtime.py              #   Live action feed, channel/measure distributions
│       ├── training.py              #   Champion vs challenger, model version timeline
│       ├── measures.py              #   Per-measure closure trend, channel effectiveness, funnel
│       ├── patient_journey.py       #   Individual patient timeline, action history, reward curve
│       ├── state_machine.py         #   Lifecycle funnel, conversion rates, transitions table
│       └── logs.py                  #   Real-time simulation log stream with level filtering
│
├── scripts/                         # CLI entry points
│   ├── generate_data.py             #   python scripts/generate_data.py --cohort-size 5000
│   ├── train_world_models.py        #   python scripts/train_world_models.py --epochs 50
│   ├── train_agent.py               #   python scripts/train_agent.py --bc-epochs 50 --cql-epochs 100
│   ├── run_simulation.py            #   python scripts/run_simulation.py --days 30
│   └── run_dashboard.py             #   python scripts/run_dashboard.py --port 8050
│
├── tests/                           # 212 tests
│   ├── conftest.py                  #   Shared fixtures (small patient cohort, all 4 datasets)
│   ├── test_datagen.py              #   Schema validation, clinical ranges, cross-dataset consistency
│   ├── test_environment.py          #   Action masking, OOD actions, reward, gym env behavior
│   ├── test_training.py             #   BC, actor-critic CQL, twin critics, evaluation
│   ├── test_simulation.py           #   Daily/nightly cycles, state machine, metrics, output formats
│   └── test_integration.py          #   E2E pipeline, world models, edge cases, dashboard data formats
│
├── data/
│   ├── generated/                   #   Mock datasets (gitignored)
│   └── simulation/                  #   Per-day simulation outputs (gitignored)
│       ├── cumulative_metrics.json  #     Dashboard reads this for all charts
│       ├── simulation_log.jsonl     #     Structured log for Logs tab
│       └── day_XX/
│           ├── actions_taken.json   #     Every action selected for every patient
│           ├── experience_buffer.json #   (obs, action, reward, mask) for retraining
│           ├── state_machine.json   #     Action lifecycle records
│           └── nightly_metrics.json #     Champion vs challenger evaluation
│
├── logs/                            #   Process logs from run.sh
│   ├── simulation.log
│   └── dashboard.log
│
├── .pids/                           #   PID files for run.sh process management
├── pyproject.toml                   #   Project metadata and dependencies
└── .gitignore                       #   Excludes data/, checkpoints/, logs/, .pids/
```
