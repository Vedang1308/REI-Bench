# Phase 2: Multi-Agent Architecture (RESOLVE)

> **Note:** This phase will be implemented after Phase 1 baselines and error analysis are complete.

## Architecture Overview

The **RESOLVE** (RE-resolution via Staged Orchestration of Language-driven Verification Experts) architecture consists of 4 specialized agents:

1. **Context Distiller** — Extracts relevant context, filters noise (ambiguous names)
2. **RE Resolver** — Resolves implicit referring expressions into explicit ones
3. **Planner** — Generates action plans using clear instructions
4. **Verifier** — Cross-checks plans against context for correctness

## Prerequisites

- Phase 1 baselines must be run first
- Error analysis from Phase 1 informs agent design choices

## Structure

```
phase2/
├── src/
│   ├── agents/
│   │   ├── re_resolver.py
│   │   ├── context_distiller.py
│   │   ├── planner.py
│   │   ├── verifier.py
│   │   └── orchestrator.py
│   ├── prompts/
│   └── pipeline.py
├── scripts/
│   ├── run_multiagent.py
│   └── compare_with_baseline.py
└── results/
```

## Usage (After Implementation)

```bash
python -m phase2.scripts.run_multiagent --model meta-llama/Llama-3.2-3B-Instruct
python -m phase2.scripts.compare_with_baseline
```
