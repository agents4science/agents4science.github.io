# Capability-Based Execution and Persistent Agent Framework for DOE Environments

## 1. Overview

This document outlines an architectural model for enabling:

1. **Capability-based remote execution** across DOE sites without requiring per-user accounts at each location.
2. **Persistent, delegated agents** that can act on behalf of users, both locally and at remote sites (e.g., near HPC systems).

The goal is to move from an **account-centric model** to a **capability- and agent-centric model**, enabling scalable, secure, and interoperable multi-site scientific workflows.

---

## 2. Problem Statement

Current DOE computing environments are largely based on:

- Per-user accounts at each site
- Interactive login and manual execution
- Ad hoc automation

This model does not scale for:

- Multi-site workflows
- Autonomous or semi-autonomous agents
- Persistent, long-running scientific processes
- Fine-grained access control

We require a model where:

- Access is granted to **capabilities**, not systems
- Execution is mediated through **policy-controlled interfaces**
- Agents can operate under **delegated authority**

---

## 3. Core Requirements

### 3.1 Capability-Based Federated Execution

Users and agents must be able to:

- Invoke executable programs or workflows at remote sites
- Without requiring full interactive accounts at those sites
- Based solely on **authorization to use specific capabilities**

### 3.2 Persistent Delegated Agency

Users must be able to:

- Launch long-lived agents
- Delegate authority to those agents
- Monitor, manage, and control agent behavior
- Run agents:
  - Locally
  - Or remotely (e.g., near HPC systems)

---

## 4. Motivating Use Cases

### 4.1 Multi-Site Simulation Campaign

#### Scenario

A materials scientist wants to screen 50,000 candidate battery electrolyte molecules. The workflow:

1. Run DFT calculations (Gaussian/VASP) at ALCF
2. Run MD simulations (LAMMPS) at NERSC
3. Run ML inference for property prediction at OLCF
4. Store results in a database at the home institution
5. Use results to select next batch of candidates (active learning loop)

The campaign runs for weeks. The scientist doesn't want to babysit it.

#### What Happens Today

- Scientist has accounts at three facilities with separate allocations
- Writes ad-hoc scripts to poll job queues, transfer data, parse outputs
- Handles failures manually (jobs timeout, nodes fail, transfers stall)
- Active learning loop requires human intervention to trigger next iteration
- If their laptop closes or VPN drops, coordination stops

#### With the Proposed Architecture

**Capabilities invoked:**

- `alcf.run_dft(molecule_spec, method, basis_set) → energies, orbitals`
- `nersc.run_md(structure, forcefield, ensemble, duration) → trajectory`
- `olcf.predict_properties(features) → predictions`
- `home.store_results(data, collection) → record_id`

**Agent behavior:**

```
Agent: electrolyte-screening-agent
Owner: researcher@anl.gov
Delegated authority: [alcf.run_dft, nersc.run_md, olcf.predict_properties, transfer:*, home.store_results]
State:
  - candidates_remaining: 47,234
  - active_jobs: {alcf: 128, nersc: 64}
  - completed: 2,766
  - budget_used: {alcf: 12,400 node-hours, nersc: 8,200 node-hours}

Loop:
  1. Select next batch from candidates (active learning model)
  2. Submit DFT jobs to ALCF
  3. On DFT completion → transfer outputs → submit MD to NERSC
  4. On MD completion → extract features → submit inference to OLCF
  5. Store results → update candidate rankings → repeat

On failure:
  - Retry with backoff
  - If repeated failure → pause and notify user

On budget threshold (80%):
  - Request human approval to continue
```

**Globus services used:**

- **Auth**: Delegation tokens scoped to specific capabilities, refreshed automatically
- **Compute**: Invoke DFT/MD/inference functions at each site
- **Transfer**: Move trajectories, checkpoints, results between sites

**What's new/needed:**

- Agent runtime that persists across weeks, maintains state, handles events
- Capability registry so agent can discover `alcf.run_dft` schema and requirements
- Policy layer: "agent can spend up to 20,000 node-hours without approval"
- Event system: job completion triggers next stage (not polling)

---

### 4.2 Knowledge-Updating Research Agent

#### Scenario

A researcher studying protein-ligand binding wants an agent that:

1. Monitors new publications (PubMed, arXiv, bioRxiv)
2. Extracts relevant binding affinity data from papers
3. Updates a local knowledge base
4. Re-trains predictive models when sufficient new data accumulates
5. Alerts the user to high-impact new findings

The agent runs indefinitely, acting as a "research assistant" that keeps the knowledge base current.

#### What Happens Today

- Manual literature review (sporadic, incomplete)
- Data extraction is labor-intensive
- Models go stale
- New relevant papers are missed

#### With the Proposed Architecture

**Capabilities invoked:**

- `arxiv.search(query, since_date) → paper_list`
- `pubmed.search(query, since_date) → paper_list`
- `llm.extract_binding_data(paper_pdf) → structured_data`
- `nersc.retrain_model(dataset, config) → model_artifact`
- `home.update_knowledge_base(records) → status`
- `notify.send_alert(user, message) → delivered`

**Agent behavior:**

```
Agent: binding-affinity-monitor
Owner: researcher@university.edu
Delegated authority: [arxiv.search, pubmed.search, llm.extract_binding_data,
                      nersc.retrain_model, home.update_knowledge_base, notify.send_alert]
State:
  - last_scan: 2026-03-29T00:00:00Z
  - knowledge_base_version: 47
  - pending_extraction: []
  - new_records_since_retrain: 127
  - model_version: 12

Schedule: daily at 02:00 UTC

Daily loop:
  1. Query arxiv + pubmed for new papers matching keywords
  2. Filter to relevant papers (LLM triage)
  3. For each relevant paper:
     a. Retrieve PDF
     b. Extract structured binding data via LLM
     c. Validate extracted data (schema check, sanity bounds)
     d. Add to pending records
  4. Update knowledge base with validated records
  5. If new_records_since_retrain > 200:
     a. Request human approval for retraining
     b. On approval: submit retrain job to NERSC
     c. On completion: update model_version
  6. If high-impact finding detected:
     a. Send alert to user with summary

On error:
  - Log and continue (don't stop agent for one bad paper)
  - If repeated failures → notify user
```

**Globus services used:**

- **Auth**: Long-lived delegation for API access, compute, storage
- **Compute**: Run extraction (LLM calls), retraining jobs
- **Transfer**: Move datasets for retraining, model artifacts back

**What's new/needed:**

- Agent runtime with **scheduled triggers** (cron-like), not just event-driven
- **External API integration** as capabilities (arxiv, pubmed)
- **Human-in-the-loop approval** workflow for expensive actions (retraining)
- **Indefinite lifespan**: agent runs for months/years, not just a campaign
- **Graceful degradation**: one bad paper shouldn't crash the agent
- **State versioning**: knowledge base and model versions tracked

---

### 4.3 Use Case Comparison

| Aspect | Simulation Campaign | Knowledge Agent |
|--------|---------------------|-----------------|
| Duration | Weeks | Indefinite |
| Trigger | Continuous (job completion) | Scheduled + events |
| Compute intensity | Very high (HPC) | Low-medium (LLM, periodic training) |
| Human interaction | Budget approvals, failure alerts | Approvals, findings alerts |
| State complexity | Job queues, candidate lists | Knowledge base, model versions |
| Failure mode | Retry jobs, pause on repeated failure | Skip bad inputs, continue |
| Sites involved | 3-4 HPC centers | APIs + 1 HPC + home storage |

### 4.4 Architectural Requirements Surfaced

These use cases reveal that the architecture must support:

1. **Both event-driven and scheduled execution**
2. **External APIs as capabilities**, not just HPC functions
3. **Human approval workflows as first-class constructs**
4. **Durable state that survives agent restarts**
5. **Budget/quota enforcement at the agent level**
6. **Different failure strategies**: aggressive retry vs. graceful degradation

---

## 5. Foundation: Globus Services

This architecture builds on existing Globus infrastructure:

- **Globus Auth**: Federated identity, OAuth2 tokens, consent management, delegation
- **Globus Compute**: Remote function execution on endpoints at DOE facilities
- **Globus Transfer**: Reliable, high-performance data movement between sites

The novel contributions of this architecture are:

1. **Capability Registry**: A discovery and metadata layer over Globus Compute endpoints, turning raw function execution into semantically meaningful, schema'd capabilities
2. **Agent Runtime**: Persistent stateful processes that orchestrate Globus services over time
3. **Policy Layer**: Finer-grained authorization than current Globus scopes—capability-level rather than endpoint-level control

---

## 6. Key Architectural Concepts

### 6.1 Capabilities

A **capability** is a named, policy-controlled action exposed by a site.

Examples:

- `submit_slurm_job`
- `run_vasp_simulation`
- `launch_screening_pipeline`
- `run_inference_model_X`
- `transfer_dataset`

Each capability includes:

- Execution mapping (executable/workflow)
- Input/output schema
- Resource constraints
- Authorization policy
- Execution mode (batch, synchronous, service)
- Auditing and provenance requirements

---

### 6.2 Capability Registry

A distributed registry that allows discovery of capabilities across sites.

Functions:

- Publish available capabilities
- Provide metadata and schemas
- Enable filtering and search
- Expose authorization requirements

---

### 6.3 Federated Identity and Delegation

Users authenticate via a federated identity system.

Instead of site accounts:

- Users receive **delegation tokens**
- Tokens assert:
  - Identity
  - Authorized capabilities
  - Constraints (scope, time, limits)

Agents may also receive delegated authority.

---

### 6.4 Site Execution Gateway

Each site provides a **gateway service** that:

- Accepts capability invocation requests
- Validates identity and authorization
- Maps requests to local infrastructure:
  - Schedulers (e.g., Slurm)
  - Containers
  - Workflow systems
- Enforces local policy
- Reports status and outputs

This isolates users from site-specific complexity.

---

### 6.5 Standard Invocation Interface

All capabilities should support a uniform interface:

- Discover capabilities
- Describe capability
- Invoke
- Monitor status
- Stream events/logs
- Cancel or modify execution
- Retrieve outputs and provenance

---

## 7. Agents

### 7.1 Definition

An **agent** is a persistent computational entity that:

- Maintains state over time
- Pursues goals
- Invokes capabilities
- Responds to events
- Acts under delegated authority

---

### 7.2 Example Agent Tasks

- Manage a large simulation campaign
- Monitor HPC job queues and resubmit failed jobs
- Trigger workflows on new data arrival
- Optimize resource usage over time
- Coordinate multi-step scientific pipelines

---

### 7.3 Agent Capabilities

Agents must be able to:

- Invoke remote capabilities
- Maintain internal state
- React to events
- Interact with users (approval, reporting)
- Adapt behavior over time

---

## 8. Agent Execution Substrate

Agents require a managed runtime environment with:

### 8.1 Persistent Identity

- Agent identity distinct from user
- Traceable delegation
- Scoped authority

---

### 8.2 Durable State

- Goal state
- Execution history
- Pending tasks
- Observations
- Credentials (securely managed)

---

### 8.3 Event Handling

Agents must respond to:

- Job completion
- Data availability
- System metrics
- External triggers
- Human approvals

---

### 8.4 Lifecycle Management

Operations include:

- Create
- Deploy
- Start / Stop
- Suspend / Resume
- Inspect
- Update
- Terminate

---

### 8.5 Observability

- Logs
- Metrics
- Traces
- Action history
- Current plan/state

---

## 9. System Architecture

### 9.1 Components

#### A. User Control Plane
- Agent creation and management
- Monitoring and approvals

#### B. Agent Runtime Plane
- Hosts agents
- Can be:
  - Local
  - Cloud
  - Site-local (near HPC/instruments)

#### C. Site Capability Gateways
- Expose executable capabilities
- Enforce site policy

#### D. Trust and Policy Plane
- Identity federation
- Authorization
- Delegation
- Auditing and provenance

---

### 9.2 Modes of Operation

#### Mode 1: Remote Execution
- Local agent invokes remote capability

#### Mode 2: Remote Agency
- Agent runs at remote site
- Directly interacts with local systems

Mode 2 improves:

- Latency
- Data locality
- Resilience
- Scheduler integration

---

## 10. Security Threat Model

This section identifies security threats specific to capability-based execution and delegated agents, along with mitigations.

### 10.1 Threat Actors

| Actor | Goals | Capabilities |
|-------|-------|--------------|
| Malicious user | Exceed allocation, access unauthorized data, disrupt others | Valid credentials, can create agents |
| Compromised agent | Exfiltrate data, consume resources, pivot to other systems | Holds delegation tokens, can invoke capabilities |
| Malicious insider (at facility) | Access user data, manipulate results | Privileged access to gateway/scheduler |
| External attacker | Steal credentials, disrupt service, cryptomining | Network access, phishing |

---

### 10.2 Threat: Token Theft and Misuse

**Attack**: Attacker obtains delegation tokens (from compromised agent, network interception, or storage breach) and invokes capabilities as the victim.

**Mitigations**:
- Short-lived access tokens (minutes) with refresh tokens stored securely
- Tokens bound to agent identity (cannot be used from different client)
- Capability invocations logged with client fingerprint
- Anomaly detection on invocation patterns
- Token revocation propagates to all gateways within seconds

---

### 10.3 Threat: Confused Deputy

**Attack**: Attacker tricks an agent into performing actions the attacker couldn't do directly. For example, a malicious input causes the agent to transfer data to an attacker-controlled location.

**Mitigations**:
- Capabilities validate that outputs go to authorized destinations only
- Agents operate with least-privilege (only capabilities needed for specific task)
- Input validation at capability boundary, not just agent
- Sensitive actions require explicit human approval regardless of agent authority

---

### 10.4 Threat: Privilege Escalation via Agent

**Attack**: User creates agent, grants it authority, then agent is manipulated (or bugs out) to perform actions beyond intended scope.

**Mitigations**:
- Delegation cannot exceed delegator's own authority
- Capability-level authorization (not blanket "run anything at ALCF")
- Budget caps enforced at gateway, not just agent
- Agents cannot modify their own authority grants
- All privilege grants logged and auditable

---

### 10.5 Threat: Resource Exhaustion

**Attack**: Malicious or buggy agent submits excessive jobs, fills storage, or otherwise exhausts shared resources.

**Mitigations**:
- Per-agent rate limits on capability invocations
- Budget thresholds trigger automatic pause + human approval
- Facilities enforce fairshare independent of agent behavior
- Kill switch: user or facility can terminate agent immediately
- Storage quotas enforced at capability level

---

### 10.6 Threat: Agent Compromise

**Attack**: Attacker gains control of agent runtime (via vulnerability, supply chain attack, or malicious agent code) and uses it to attack facilities or exfiltrate data.

**Mitigations**:
- Agent runtime sandboxed (containers, VMs, or managed service)
- Agents cannot access each other's state or credentials
- Network egress restricted to known capability endpoints
- Agent code reviewed/signed before deployment (for high-privilege agents)
- Behavioral monitoring: unexpected capability patterns trigger alerts

---

### 10.7 Threat: Capability Abuse

**Attack**: Legitimate user uses capabilities for unintended purposes (e.g., cryptomining via `run_simulation` capability).

**Mitigations**:
- Capabilities are specific, not general-purpose shell access
- Resource usage profiling detects anomalous patterns
- Facilities can inspect job contents for high-resource requests
- Terms of service with enforcement mechanisms
- Allocation charging makes abuse expensive

---

### 10.8 Threat: Data Exfiltration

**Attack**: Agent (malicious or compromised) transfers sensitive data to unauthorized locations.

**Mitigations**:
- Transfer destinations restricted by policy (e.g., only to user's authorized endpoints)
- Large transfers require human approval
- Data classification labels enforced at capability level
- Egress monitoring and alerting
- Capabilities cannot create new Globus endpoints

---

### 10.9 Threat: Stale or Orphaned Agents

**Attack**: User leaves institution; their agents continue running with outdated authority, potentially accessing resources they should no longer have.

**Mitigations**:
- Delegation tied to user's identity provider status (IdP deprovisioning revokes tokens)
- Maximum agent lifetime with mandatory re-authorization
- Periodic attestation: user must confirm agent should continue
- Facility can enumerate and terminate agents for deprovisioned users

---

### 10.10 Threat: Cross-Site Attack Propagation

**Attack**: Compromised agent at one facility uses its authority to attack another facility in a multi-site workflow.

**Mitigations**:
- Per-site authorization (token for ALCF doesn't grant NERSC access)
- Each facility validates independently; no transitive trust
- Cross-site workflows logged end-to-end for forensic analysis
- Facilities can block specific agents without blocking the user entirely

---

### 10.11 Incident Response

When a security incident is detected:

1. **Immediate**: Revoke agent's delegation tokens (propagates to all sites)
2. **Contain**: Terminate running jobs submitted by the agent
3. **Investigate**: Audit logs identify all capability invocations
4. **Notify**: Alert user, affected facilities, and (if required) security teams
5. **Remediate**: Patch vulnerability, rotate credentials if needed
6. **Review**: Post-incident analysis to improve defenses

---

### 10.12 Security Invariants

The architecture must maintain these invariants:

1. **No authority amplification**: Agents cannot gain more authority than explicitly granted
2. **Traceable actions**: Every capability invocation attributable to user + agent
3. **Revocable access**: Any delegation can be revoked within minutes
4. **Site sovereignty**: Facilities can deny any request regardless of valid tokens
5. **Fail secure**: Token validation failures result in denial, not access

---

## 11. Policy Controls

### 11.1 Delegation Constraints

Delegation must be:

- Scope-limited (specific capabilities, not blanket access)
- Time-limited (explicit expiration)
- Revocable (immediate effect)
- Auditable (all grants logged)

### 11.2 Site Control

Sites retain authority over:

- Which capabilities are exposed
- Resource limits per capability/user/agent
- Execution constraints
- Policy enforcement and override

### 11.3 Resource and Cost Control

Agents must operate under:

- Quotas (invocations, compute hours, storage)
- Budget constraints (with approval thresholds)
- Rate limits
- Kill switches (user and facility level)

### 11.4 Audit and Provenance

All actions must be:

- Logged with timestamps
- Traceable to user and agent
- Reproducible (inputs recorded)
- Retained per compliance requirements

---

## 12. Minimal API Surface

### 11.1 Capability API

- `discover_capabilities(site, filters)`
- `describe_capability(capability_id)`
- `invoke(capability_id, inputs, delegation_token)`
- `get_run_status(run_id)`
- `stream_run_events(run_id)`
- `cancel_run(run_id)`
- `fetch_outputs(run_id)`

---

### 11.2 Agent API

- `create_agent(spec, owner, policy)`
- `deploy_agent(runtime)`
- `start_agent(agent_id)`
- `stop_agent(agent_id)`
- `inspect_agent(agent_id)`
- `grant_authority(agent_id, scope, ttl)`
- `revoke_authority(agent_id)`
- `list_agent_actions(agent_id)`
- `approve_action(agent_id, action_id)`

---

## 12. Advantages Over Account-Based Models

- Fine-grained access control (capability-level)
- Reduced user friction
- Improved auditability
- Better support for automation and agents
- Multi-site interoperability
- Clear separation of policy and execution

---

## 13. Architectural Thesis

> Scientific computing infrastructure should evolve from account-based access to capability-based execution, and from user-driven workflows to persistent, delegated agents operating under policy-constrained authority across distributed environments.

---

## 14. Implications for DOE Facilities

Adopting this architecture has specific implications for leadership computing facilities (ALCF, OLCF, NERSC) and other DOE sites.

### 14.1 What Facilities Must Provide

#### Capability Gateways

Each facility deploys a gateway service (building on Globus Compute endpoints) that:

- Exposes site-approved capabilities with defined schemas
- Maps capability invocations to local schedulers (Slurm, PBS)
- Enforces site-specific policies (allowed users, resource limits, permitted software)
- Handles credential translation (Globus Auth tokens → local execution context)

#### Capability Definitions

Facilities must define and publish capabilities for their resources:

| Facility | Example Capabilities |
|----------|---------------------|
| ALCF | `aurora.run_dft`, `aurora.run_lammps`, `aurora.inference_service` |
| OLCF | `frontier.run_vasp`, `frontier.gpu_inference`, `frontier.workflow_submit` |
| NERSC | `perlmutter.run_quantum_espresso`, `perlmutter.jupyter_spawn`, `perlmutter.data_analysis` |

Each capability includes:
- Input/output schema (JSON Schema or similar)
- Resource requirements and constraints
- Authorization requirements (allocation, project membership)
- Execution semantics (batch, interactive, service)

#### Integration with Allocation Systems

Capabilities must integrate with existing allocation mechanisms:

- **ERCAP/INCITE/ALCC**: Capability invocations charged against project allocations
- **Fairshare**: Agent-submitted jobs subject to standard queue policies
- **Quotas**: Agents operate within allocation limits; exceeding triggers approval workflows

### 14.2 What Facilities Retain Control Over

Sites maintain full authority over:

- **Which capabilities to expose**: No obligation to expose everything
- **Who can invoke capabilities**: Can require allocation, project membership, or explicit approval
- **Resource limits**: Per-capability, per-user, per-agent constraints
- **Software stack**: Capabilities run in site-controlled environments
- **Security policy**: Sites can revoke access, audit usage, require additional authentication

### 14.3 Benefits to Facilities

#### Reduced Account Management Burden

- Users invoke capabilities without full interactive accounts
- No per-user home directories or shell access required for many use cases
- Simplified onboarding for collaborators

#### Better Resource Utilization

- Agents can optimize job submission (backfill, preemption-tolerant jobs)
- Cross-site workflows use resources where available
- Automated retry reduces wasted allocation due to transient failures

#### Improved Auditability

- All actions traceable to specific users and agents
- Capability-level logging vs. raw system calls
- Clear provenance for scientific reproducibility

#### Gradual Adoption Path

Facilities can adopt incrementally:

1. **Phase 1**: Expose a few high-value capabilities (e.g., standard simulation codes)
2. **Phase 2**: Add capability discovery and schema publication
3. **Phase 3**: Support agent-initiated invocations with delegation
4. **Phase 4**: Host site-local agent runtimes for low-latency workflows

### 14.4 Coordination Requirements

Facilities must coordinate on:

- **Common capability schema format** (so agents can discover and invoke across sites)
- **Interoperable authorization** (Globus Auth as common trust anchor)
- **Event notification standards** (job completion, data availability)
- **Cross-site accounting** (for multi-facility campaigns)

### 14.5 Example: Multi-Site Simulation at ALCF + NERSC

```
User: researcher@university.edu
Agent: electrolyte-screening-agent
Allocation: INCITE project CHM123

1. Agent invokes alcf.run_dft(molecule_spec)
   - Gateway validates: user has CHM123 allocation, capability is authorized
   - Maps to: sbatch on Aurora with project account
   - Charges: 100 node-hours to CHM123

2. On completion, agent receives event notification
   - Invokes Globus Transfer: Aurora → Perlmutter

3. Agent invokes nersc.run_md(trajectory)
   - Gateway validates: user has NERSC allocation (separate from ALCF)
   - Maps to: sbatch on Perlmutter
   - Charges: 50 node-hours to NERSC allocation

4. Results stored, agent updates state, selects next batch
```

Both facilities see:
- Which user/agent made the request
- Which capability was invoked
- Resources consumed
- Full audit trail

---

## 15. Next Steps

### 15.1 Infrastructure Development

1. **Capability schema specification**: Define JSON Schema-based format for capability metadata, inputs, outputs, and constraints
2. **Capability registry service**: Build discovery layer over Globus Compute endpoints
3. **Reference site gateway**: Implement gateway for one facility (e.g., ALCF) with Slurm integration
4. **Agent runtime prototype**: Build persistent agent execution environment with state management

### 15.2 Demonstration Use Cases

**Use Case A: Multi-Site Simulation Campaign**
- Demonstrate electrolyte screening workflow across ALCF, NERSC, OLCF
- Show automated failure recovery and job resubmission
- Implement budget-based human approval workflow
- Measure: time-to-completion vs. manual orchestration

**Use Case B: Knowledge-Updating Research Agent**
- Deploy literature monitoring agent for a research group
- Demonstrate scheduled execution, external API integration
- Show graceful degradation on extraction failures
- Measure: coverage and latency vs. manual literature review

### 15.3 Facility Engagement

1. Engage ALCF, NERSC, OLCF on capability gateway requirements
2. Align with existing Globus Compute deployments at facilities
3. Coordinate on cross-site authorization and accounting
4. Identify pilot user communities for early adoption

### 15.4 Policy and Governance

1. Define delegation constraint language and enforcement
2. Establish cross-facility coordination on capability standards
3. Develop audit and compliance requirements
4. Create incident response procedures for agent misbehavior

---

## 16. Summary

This model unifies:

- Secure remote execution
- Persistent automation
- Multi-site interoperability
- Policy-driven control

It provides a foundation for **agentic science workflows at DOE scale**.
