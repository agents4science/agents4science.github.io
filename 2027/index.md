# AI Agents for Science --- Winter 2027

*CMSC 35370, The University of Chicago. Instructor: Ian Foster. Winter quarter 2027 (9 weeks, early January to mid-March).*
<!-- TODO: confirm course number, meeting times, room, TA, office hours -->

The 2027 edition of [AI Agents for Science](../Class/) is rebuilt around a single principle: **you learn agents by running them, breaking them, and fixing them**. Lectures are compressed into short briefings; most class time is lab and studio. Every student ships something every week, and the class itself operates as an instrumented agent deployment: every student receives a token budget with a visible governor, agents act through sandboxed tools, and the instructors supervise the fleet through an oversight dashboard. When your agent exhausts its budget or gets hijacked by a malicious web page, that is not an inconvenience---it is the curriculum.

The course serves both **computer scientists** (how do we build and govern these systems?) and **domain scientists** (what can they do for my science, and when should I trust them?). No prior agent experience is required; comfort with Python and the command line is assumed.

## Format

Two 80-minute sessions per week. Each session opens with a briefing of at most 20 minutes; the remainder is hands-on lab and studio work.

Three recurring rituals:

- **Bake-offs.** Several times during the quarter, everyone's agent attempts the same scientific task on a shared harness, with results on a class leaderboard. Graded on participation and writeup, not rank.
- **Failure autopsies.** Each week, one student presents their *worst* agent trace of the week and the class diagnoses it together. Failure is data here.
- **Red-team week.** In week 6, you attack each other's agents with adversarially crafted content, then measure which defenses actually work.

## The project

The spine of the course is the **reproduction challenge**: build an agent that attempts end-to-end computational reproduction of results from published papers (paper + data + code in; reproduction report + full execution trace out). Reproducibility is a real scientific problem, the task is unambiguous, and every hard issue in agentic AI shows up on the way: cost control, tool failures, provenance, knowing when to stop.

In the final three weeks, teams either push the reproduction challenge further or take on a **client project**: a concrete task posed by a research group at UChicago or Argonne that wants it done. Client projects are vetted before the quarter begins: each client commits a one-page task specification, working data access, and demo-day attendance. Domain-science students may propose their own group's problem as a client project under the same terms. Teams without a suitable client pursue the reproduction challenge; nobody's grade depends on a third party's responsiveness. Prospective clients: See the [call for client projects](clients.md).
<!-- TODO: recruit and confirm 4-6 client groups (target: commitments by mid-December) -->

## Schedule

| Week | Briefing themes | Lab / studio | Deliverable |
|---|---|---|---|
| 1 | What is an agent? From models to agents | Build a bare agent loop from scratch (no framework, ~100 lines) | Working tool-calling agent |
| 2 | Tools, environments, MCP | Add search and sandboxed code execution; your traces are captured from day one | Instrumented agent; **Bake-off 1:** data extraction |
| 3 | Memory, persistence, budgets | Add memory; meet your budget governor | Agent that survives budget pressure; **Autopsy 1** |
| 4 | Literature agents; RAG | Synthesis with verified citations | **Bake-off 2:** evidence gathering, claims checked |
| 5 | Data and code agents | Sandboxed analysis of real datasets | **Bake-off 3:** reproduce one figure from a published paper |
| 6 | Security: control--data ambiguity, enforcement | **Red-team week:** attack week-5 agents; compare prompt guardrails vs. sandbox enforcement | Attack + defense writeup |
| 7 | Multi-agent systems and delegation; project launch | Project sprint 1 | Project plan + first traces |
| 8 | Agents for scientific facilities; evaluation | Project sprint 2; standups | Working prototype |
| 9 | What agents mean for computer science | **Demo day** | Demo, report, and your best + worst trace |

Readings each week pair one foundational paper with one current one; the list builds on the [2026 curriculum](../Class/curriculum.md) and the instructor's article *Agency as a New Organizing Abstraction for Computer Science* (CACM, 2026).
<!-- TODO: per-week reading list; add arXiv link when posted -->

## Grading

Weekly labs and bake-off writeups 40%; project (plan, prototype, demo, report) 40%; autopsy presentation and participation 20%. Using agents to do the work is not cheating---it is the point---but you must disclose how, and you are responsible for everything you submit.

## Infrastructure

All students work through a course gateway that provides per-student model access, token budgets, and automatic trace capture, with sandboxed tool execution and a shared bake-off harness. The components are described in the [instrument development plan](instruments.md); they build on the group's [Bursar](../Bursar/), [Resource Rights](../ResourceRights/), and [agentic capabilities](../Capabilities/) work. Token budgets are provided; you will not pay for API access.

## Research participation (optional)

The course is also the pilot deployment for an open research program on how agentic systems fail and how they can be governed. Students may *optionally* donate their de-identified agent traces to an open research corpus. Participation is opt-in, fully decoupled from grades, and governed by an IRB protocol.
<!-- TODO: IRB protocol number and consent link once approved -->

---

*Previous edition: [Autumn 2025](../Class/).*
