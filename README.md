# react-retry-waste-analysis
ReAct agents waste 92.6% of retries. Here's the architecture fix (error taxonomy + circuit breakers + deterministic routing) that drops waste to 0%.

# ReAct Retry Waste Analysis

> **90.8% of retries in a standard ReAct agent are wasted on errors that can never succeed.**  
> This repo contains the benchmark that found it — and the three structural fixes that eliminate it.

📄 **Article:** [Your ReAct Agent Is Wasting 90% of Its Retries — And You Don't Even See It](https://towardsdatascience.com/your-react-agent-is-wasting-90-of-its-retries-heres-how-to-stop-it/) *(Towards Data Science)*

---

## What This Is

A deterministic, fully reproducible simulation comparing two agent architectures across 200 tasks:

- **ReAct Agent** — standard Thought → Action → Observation loop with a global retry counter
- **Controlled Workflow** — deterministic plan execution with error taxonomy, per-tool circuit breakers, and typed tool routing

The single architectural difference: **where tool names are resolved** — from LLM output (ReAct) or from a Python dict at plan time (Workflow).

---

## Quick Start

```bash
git clone https://github.com/Emmimal/react-retry-waste-analysis
cd react-retry-waste-analysis
pip install matplotlib   # only dependency beyond stdlib
python app.py --seed 42
```

Every number in the article is reproduced exactly by `--seed 42`.

---

## Key Results (seed=42, 200 tasks)

| Metric | ReAct | Workflow |
|---|---|---|
| Success rate | 89.5% | 100.0% |
| Total retries | 513 | 80 |
| Wasted retries | **466 (90.8%)** | **0 (0.0%)** |
| Hallucination events | 155 | 0 |
| Step σ | 1.36 | 0.46 |
| P95 latency (ms) | 143.3 | 146.2 |
| Estimated cost ($) | $0.3450 | $0.3222 |

---

## CLI Options

```bash
# Full 200-task benchmark
python app.py --seed 42

# Watch a single task execute in verbose mode
python app.py --replay 7

# Export results to JSON
python app.py --seed 42 --export-json

# Custom task count or seed
python app.py --tasks 500 --seed 99

# Skip plot generation
python app.py --no-plots

# Custom plot output directory
python app.py --plot-dir my_plots
```

---

## Output

Running the benchmark produces:

- Full results table (success rate, retry budget, error taxonomy, latency, cost)
- Sensitivity analysis at hallucination rates of 5%, 15%, and 28%
- 6 figures saved to `plots/`:
  - `fig1_success_hallucinations.png`
  - `fig2_retry_budget.png`
  - `fig3_step_distribution.png`
  - `fig4_error_taxonomy.png`
  - `fig5_latency_cdf.png`
  - `fig6_sensitivity.png`

---

## The Three Fixes

### Fix 1 — Error Taxonomy
Classify errors at the point they're raised. Non-retryable errors (`TOOL_NOT_FOUND`, `INVALID_INPUT`) emit `RETRY_SKIPPED` and consume zero budget. Applicable to a ReAct agent without changing its architecture.

### Fix 2 — Per-Tool Circuit Breakers
Each tool gets its own `CircuitBreaker` instance. A degraded tool fails fast without draining budget for other tools.

### Fix 3 — Deterministic Tool Routing *(the structural differentiator)*
Tool names are resolved from `STEP_TO_TOOL: dict[StepKind, str]` at plan time — never from LLM output. Hallucination at the routing layer becomes structurally impossible.

---

## Simulation Parameters

| Parameter | Value | Notes |
|---|---|---|
| `SEED` | 42 | Global random seed |
| `NUM_TASKS` | 200 | Tasks per experiment |
| `hallucination_rate` | 28% | Conservative estimate from published benchmarks |
| `HALLUCINATION_RETRY_BURN` | 3 | Retry slots burned per hallucination event |
| `MAX_REACT_RETRIES` | 6 | Global retry budget for ReAct |
| `SENSITIVITY_RATES` | 5%, 15%, 28% | Hallucination rates for sensitivity sweep |

**Note:** The 28% hallucination rate is a calibrated parameter, not a directly reported figure. Your observed rate will vary with model, prompt quality, and tool schema design.

---

## Limitations

- Latency figures are simulated — do not use for capacity planning
- `HALLUCINATION_RETRY_BURN = 3` influences the exact waste percentage; the structural conclusion (workflow wastes 0%) holds at all values
- The workflow's zero hallucination count is a simulation design property; hallucinations can still occur upstream of routing in production
- Three tools is a simplified environment; threshold values will need tuning for your workload

---

## References

- Yao et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023. [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
- Shinn et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning.* NeurIPS 2023. [arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366)
- Fowler, M. (2014). *CircuitBreaker.* [martinfowler.com](https://martinfowler.com/bliki/CircuitBreaker.html)

---

## Requirements

- Python 3.9+
- `matplotlib` (plots only — all other dependencies are stdlib)

---

## License

MIT
