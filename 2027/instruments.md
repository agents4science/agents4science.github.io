# Instrument Development Plan --- Winter 2027 Course

*The software that runs [the course](index.md), and doubles as reference implementations for an open research program on governing agentic systems.*

The course operates as an instrumented agent deployment: every student agent runs against a budget, acts through sandboxed tools, leaves a complete trace, and is supervisable from an instructor dashboard. These are the same mechanisms the research program studies (budget governance, enforcement placement, failure attribution, oversight at scale), so each course instrument is also a research deliverable.

## Instruments

**I1. Course gateway (keystone; new build).** A model-access proxy through which all student agents call LLM APIs: per-student keys, token metering, hard and soft budget caps, complete request/response logging, and model-agnostic routing. One component provides budgets, traces, and provider independence at once; everything else attaches to it. Build on an existing open-source LLM proxy rather than from scratch.

**I2. Budget governor (adapt [Bursar](../Bursar/)).** Policy layer on the gateway: Allocations, burst behavior, back pressure, and a per-student usage dashboard. Course role: Week-3 lab and quarter-long cost control. Research role: Reference budget-governance implementation.

**I3. Tool sandbox and permissions (adapt [Resource Rights](../ResourceRights/)).** Containerized code execution plus capability-scoped tool access mediated at the tool layer (MCP), so an agent's permissions are set by policy, not prompt. Course role: All labs from week 2; the week-6 enforcement-placement experiment. Research role: Reference enforcement mechanism.

**I4. Trace store and viewer.** A common trace schema (every model call, tool call, and observation, with costs), durable storage keyed by student and session, and a web viewer good enough to scrub through a session in class. Course role: Weekly autopsies. Research role: The atlas ingestion format.

**I5. Oversight dashboard.** Instructor view over the fleet: Live usage, recent sessions, budget and anomaly flags, escalations. Version 0 is usage plus trace links; refinement during the quarter is itself an oversight-design exercise.

**I6. Bake-off harness and leaderboard.** Task specification, runner, scoring, and a class leaderboard, built on existing open evaluation tooling. Course role: Bake-offs 1--3 and the reproduction challenge. Research role: The fidelity-tier environment pattern.

**I7. Consent and redaction module (phase 2, with IRB).** Opt-in flag at the gateway; de-identification at ingest; export path to the open corpus. Built during the quarter; nothing is exported before IRB approval.

**I8. Labeling pipeline (phase 2).** LLM judges plus a human-validation interface, seeded by autopsy sessions. Built during weeks 3--6, once real traces exist.

## Timeline (October 2026 → March 2027)

| Sprint | Through | Goal |
|---|---|---|
| 0 | late Oct | Decisions: Proxy base, trace schema, provider(s), sandbox approach. Repo, CI, and skeleton. |
| 1 | mid Nov | **I1 MVP:** Gateway with per-student keys, hard caps, full logging. |
| 2 | end Nov | **I3 + I4 v0:** Sandboxed execution, lab tool set, trace viewer. |
| 3 | mid Dec | **I2, I5, I6 v0:** Governor policies, instructor dashboard, bake-off harness; week-6 red-team content pack. |
| Dry run | late Dec | TAs run weeks 1--2 labs end to end on the production stack. |
| Freeze | Jan 2 | Tag v1.0. During the quarter: Fixes only on I1--I6; build I7--I8. |

**Priorities.** P0 (class cannot run without): I1, I3, I4. P1: I2, I5, I6. P2: I7, I8.

**Staffing.** One RSE or TA plus instructor; coding agents are used throughout for implementation, with the development itself instrumented---the toolchain eating its own output is part of the research program.

**Risks.** (1) Token provisioning logistics (per-student access under an institutional or sponsored account) --- resolve in sprint 0. (2) IRB timing --- submit now; the course runs regardless, since I7 gates only the research export, not the class. (3) Scope creep --- the gateway is the only novel P0 component; everything else adapts existing code or tooling.

## Resources (assuming 30 students)

**Tokens.** Budget roughly 50M tokens per student for the quarter (weekly labs at 2--5M tokens, bake-off bursts, and a long-horizon project tail), enforced by the governor. With ~30% overhead for TA and instructor use, the December dry run, red-team sweeps, LLM judges, and development itself: **2--3B tokens total**, roughly \$10--20K at blended list prices (\$300--500 list-equivalent per student). Hold a 30% reserve unallocated; running dry in week 7 is the most demoralizing failure available. Procurement, not amount, is the risk: Sponsored education credits or a purchase order must be initiated in sprint 0.

**People.** TA-1 (infrastructure): The committed engineering quarter---but note the build happens in November--December, before winter TA appointments begin, so the autumn gap (~2 person-months) needs RA, RSE, or hourly funding. TA-2 (course operations): At 30 students with weekly deliverables, one person cannot both run infrastructure and grade; a second TA or grader is strongly recommended. Instructor plus two TAs gives a 1:10 ratio during studio time, about the floor for hands-on teaching.

**Compute and hosting.** One VM or small campus Kubernetes namespace for gateway, trace store, and dashboards (traces for 3B tokens are tens of GB---trivial). Sandboxed execution is bursty: A couple of servers or a campus research-computing allocation. Under \$1--2K incremental.

**Content.** The reproduction paper set (15--30 papers with code and data, license-checked, *pre-tested by TAs for feasibility*---2--3 weeks of curation; artifact-badged papers are the shortcut); the week-6 red-team pack; 4--6 client commitments by mid-December; the IRB protocol (effort, not money, but on the critical path now).

| Item | Estimate | Likely source |
|---|---|---|
| Tokens (2--3B) | \$10--20K list | Sponsored credits |
| TA-2 (winter) | 1 TA quarter | Department |
| Autumn engineering (~2 person-months) | \$15--25K loaded | RA/RSE/hourly |
| Hosting + sandbox compute | <\$1--2K | Campus/lab in-kind |

## Mapping to the research program

| Instrument | Course use | Research thrust |
|---|---|---|
| I1 gateway + I4 traces | All labs, autopsies | T1 failure atlas (ingestion + lab arm) |
| I2 governor | Week 3, cost control | T3 anticipatory budgets |
| I3 sandbox/permissions | Week 6 red team | T4 enforcement placement |
| I5 dashboard | Fleet supervision | T7 oversight economics |
| I6 harness | Bake-offs, reproduction challenge | T2 continuum environments |
| I7 + I8 | Trace donation, autopsy labels | T1 field pilot + labeling pipeline |
