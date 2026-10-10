# The Default Project: The Reproduction Challenge

*[AI Agents for Science](index.md), Winter 2027. This is the project every team pursues unless they take on a vetted [client project](clients.md); it is also the fallback if a client project falls through.*

## The task

In weeks 7--9, your team of three builds a **reusable reproduction agent**---not a one-off script---and runs it against a portfolio of published papers, attempting end-to-end computational reproduction: Obtain the code and data, set up the environment, run the analysis, and compare outputs against the published figures and tables. Papers come from a curated set that the teaching staff has license-checked and pre-tested for feasibility, so no team's quarter dies to a broken artifact.

Everyone piloted this in week 5 (Bake-off 3: Reproduce one figure). The project is the full version: More papers, a real agent, and serious analysis.

## Two tracks

- **Breadth.** 3--5 papers, reproduced as far as the agent can get. Stresses robustness, generalization across repositories, and cost control.
- **Depth.** One genuinely hard paper, end to end. Stresses persistence, environment wrangling, and judgment about when to stop.

Ambitious extensions for either track: Apply a paper's method to a new dataset, or document an actual error in a published analysis.

## Deliverables

1. **The agent**, running on the course infrastructure.
2. **A reproduction report per paper**: Reproduced, partially reproduced, or failed---with a *diagnosis* (missing dependency, undocumented parameter, data drift, genuine discrepancy, ...).
3. **Complete execution traces** for all attempts.
4. **A self-analysis**: Where your token budget went, where the agent got stuck, which failure modes you observed---the course's themes applied to your own system.

## How you are graded

On the quality of the agent, the rigor of the reports, and the correctness of the diagnoses---**not** on whether the papers reproduce, which is not in your control. A well-documented failed reproduction with a correct diagnosis is a full-credit result. (It is also, scientifically, often the most valuable one.)

## Demo day

Your agent attempts a **held-out paper it has never seen, live**. Nobody in the room, including you, knows whether it will succeed. That is the honest test of whether you built an agent or a script.

## Why this project

The inputs are unambiguous, results are comparable across teams, and every hard problem in agentic AI appears along the way: Budgets, provenance, tool failures, knowing when to stop. The outputs are real contributions---collectively, the class produces a small open corpus of reproduction reports and agent traces that feeds the associated research program on how agentic systems fail.
