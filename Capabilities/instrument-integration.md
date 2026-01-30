# Instrument Integration

Agents that orchestrate data collection from scientific instruments—beamlines, microscopes, spectrometers, and lab automation systems.

---

## Overview

| Aspect | Details |
|--------|---------|
| **Task** | Agent controls instruments and responds to real-time data |
| **Approach** | Academy + facility instrument APIs (EPICS, Bluesky, etc.) |
| **Status** | <span style="color: blue;">**Emerging**</span> |
| **Scale** | 1 agent + instrument cluster |

---

## When to Use

- Autonomous experimental campaigns at user facilities
- Adaptive measurements that respond to sample characteristics
- Multi-instrument coordination
- 24/7 operation with minimal human intervention

---

## Architecture

```
+──────────────────+         +─────────────────────────────+
│   Experiment     │         │      Instrument Cluster      │
│   Agent          │         │                              │
│  +─────────+     │ control │  +─────────+  +─────────+   │
│  │  Agent  │─────┼────────>│  │Beamline │  │Detector │   │
│  │         │     │         │  +─────────+  +─────────+   │
│  │         │<────┼─────────┼──+─────────+  +─────────+   │
│  +────┬────+     │  data   │  │ Sample  │  │  Robot  │   │
│       │          │         │  │ Stage   │  │         │   │
│       v          │         │  +─────────+  +─────────+   │
│  +─────────+     │         │                              │
│  │   LLM   │     │         +─────────────────────────────+
│  +─────────+     │
+──────────────────+
```

---

## Capabilities

### Adaptive Measurement
Agent analyzes incoming data and adjusts measurement parameters:
- Exposure time based on signal strength
- Scan resolution based on feature detection
- Sample position to track regions of interest

### Sample Triage
Agent evaluates samples and prioritizes promising candidates:
- Quick screening measurements
- Detailed characterization of interesting samples
- Automatic rejection of poor-quality samples

### Multi-Instrument Coordination
Agent orchestrates workflows across multiple instruments:
- Sample preparation -> measurement -> analysis
- Parallel measurements on different instruments
- Data fusion from complementary techniques

---

## Example: Autonomous Beamline

```python
@tool
def move_sample(x: float, y: float):
    """Move sample stage to specified position."""
    bluesky.run(mv(sample.x, x, sample.y, y))

@tool
def collect_diffraction(exposure: float) -> np.ndarray:
    """Collect X-ray diffraction pattern."""
    return bluesky.run(count([detector], exposure))

@tool
def analyze_pattern(data: np.ndarray) -> dict:
    """Analyze diffraction pattern for crystal quality."""
    return analysis_service.process(data)

# Agent runs autonomous experiment
agent_prompt = """
Survey the sample grid. For each position:
1. Collect a quick diffraction pattern (1s exposure)
2. If crystalline peaks detected, collect high-quality pattern (30s)
3. Move to next position
Report the best crystal locations when complete.
"""
```

---

## Supported Facilities

| Facility | Instruments | Status |
|----------|-------------|--------|
| APS (ANL) | Beamlines | Pilot projects |
| NSLS-II (BNL) | Beamlines | Planned |
| LCLS (SLAC) | X-ray laser | Planned |

---

## Getting Started

Contact the CAF team for information on pilot projects at DOE user facilities.
