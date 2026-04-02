# react-retry-waste-analysis
ReAct agents waste 92.6% of retries. Here's the architecture fix (error taxonomy + circuit breakers + deterministic routing) that drops waste to 0%.

# ReAct Agents Waste 92.6% of Their Retries — The Fix Isn’t in the Model

A 200-task production simulation that exposes why most retries in ReAct-style agents are completely wasted and how a better architecture (error taxonomy + circuit breakers + deterministic routing) fixes it.

### Key Findings
- **92.6%** of retries in ReAct were wasted on non-retryable errors (mainly hallucinated tools)
- Controlled Workflow: **0%** wasted retries
- Step predictability improved dramatically (σ reduced from **1.41 → 0.46**)
- Hallucinations became structurally impossible

## Features
- Fully reproducible experiment (`python app.py --seed 42`)
- Error taxonomy with `RETRY_SKIPPED` events
- Per-tool circuit breakers
- Sensitivity analysis (5%, 15%, 28% hallucination rates)
- 6 publication-ready matplotlib figures
- Structured logging + cost tracking

## Quick Start

```bash
git clone https://github.com/Emmimal/react-retry-waste-analysis.git
cd react-retry-waste-analysis
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py --seed 42
```

## Replay a single run verbosely
```
python app.py --replay 42
```
## Files

- app.py — Complete self-contained script (Python 3.9+)
- plots/ — All 6 figures generated automatically
- experiment_results.json — Full structured results

## Article
Read the full story: ReAct Agents Waste 92.6% of Their Retries — The Fix Isn’t in the Model 
## License
MIT License — feel free to use, modify, and learn from the code.

